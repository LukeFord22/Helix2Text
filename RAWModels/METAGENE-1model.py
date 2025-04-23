import os
import requests
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import wandb
import random
import time
from rouge_score import rouge_scorer
from torch.cuda.amp import autocast, GradScaler

#=======================================================================================================
#                                                                                   Adjustable Parameters
#=======================================================================================================
torch.autograd.set_detect_anomaly(True)

# Add early stopping parameters
EARLY_STOP_PATIENCE = 3
best_val_f1 = 0.3

#Run parameters
BATCH_SIZE = 2
NUM_EPOCHS = 8
LEARNING_RATE = 3e-5
MAX_SEQ_LENGTH = 512
GRAD_ACCUM_STEPS = 5
TRAIN_DATA_FRACTION = 1
SEMANTIC_LOSS_WEIGHT = 0.5
FROZEN_LAYERS = 0

#=======================================================================================================
#                                                                                         Model loading
#=======================================================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load METAGENE-1-BnB-4Bit model with quantization
dna_model_path = "metagene-ai/METAGENE-1-BnB-4Bit"
dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_path, trust_remote_code=True)
dna_model = AutoModelForCausalLM.from_pretrained(
    dna_model_path,
    trust_remote_code=True, # Use the new quantization configuration
    device_map="auto"  # Automatically map layers to GPU if available
)


#== BioGPT Model ==
text_model_path = "microsoft/biogpt-large-pubmedqa"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_path, trust_remote_code=True)
if text_tokenizer.pad_token is None:
    text_tokenizer.pad_token = text_tokenizer.eos_token
    text_tokenizer.pad_token_id = text_tokenizer.eos_token_id
text_tokenizer.padding_side = 'left'

# Remove device_map="auto" and use regular device placement instead
text_model = AutoModelForCausalLM.from_pretrained(
    text_model_path, 
    trust_remote_code=True,
    pad_token_id=text_tokenizer.eos_token_id
).to(device) 

def freeze_gpt_layers(model, num_layers_to_freeze=FROZEN_LAYERS):
    for i, block in enumerate(model.gpt_cross.transformer_blocks):
        if i < num_layers_to_freeze:
            for param in block.parameters():
                param.requires_grad = False
    print(f"✅ Frozen BioGPT layers: 0 through {num_layers_to_freeze - 1}")


#=======================================================================================================
#                                                                                        Dataset Loading
#======================================================================================================= 

def download_dataset(url, save_path):
    """Download the dataset if it doesn't exist locally."""
    if not os.path.exists(save_path):
        print(f"Downloading dataset to {save_path}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete!")
    else:
        print(f"Dataset already exists at {save_path}")

# GitHub release asset URL
dataset_url = "https://github.com/LukeFord22/Helix2Text/releases/download/v0.1-data/protein_to_dna_sequence_descriptions.csv"
local_path = "protein_to_dna_sequence_descriptions.csv"

# Download if needed
download_dataset(dataset_url, local_path)

# Load the data
df = pd.read_csv(local_path)

sampled_df = df.sample(frac=TRAIN_DATA_FRACTION, random_state=42)
train_df = sampled_df.sample(frac=0.9, random_state=42)
test_df = sampled_df.drop(train_df.index)

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
})

def tokenize_fn(example):
    dna_enc = dna_tokenizer(
        example["DNA_Sequence"], 
        truncation=True, padding="max_length", max_length=MAX_SEQ_LENGTH
    )
    text_enc = text_tokenizer(
        example["text"], 
        truncation=True, padding="max_length", max_length=MAX_SEQ_LENGTH
    )
    return {
        "dna_input_ids": dna_enc["input_ids"],
        "dna_attention_mask": dna_enc["attention_mask"],
        "text_input_ids": text_enc["input_ids"],
        "text_attention_mask": text_enc["attention_mask"],
    }

tokenized_data = dataset.map(tokenize_fn, batched=True)
tokenized_data.set_format("torch")

sample_data = tokenized_data['train'].select(range(10))
sample_df = pd.DataFrame(sample_data)

train_loader = DataLoader(tokenized_data['train'], batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(tokenized_data['test'], batch_size=BATCH_SIZE, pin_memory=True)


#=======================================================================================================
#                                                                                     Model Architecture
#=======================================================================================================

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Fusion Model

class NormalizedGatedCrossAttentionBlock(nn.Module):
    def __init__(self, dim, cross_dim, n_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, kdim=cross_dim, vdim=cross_dim,
            num_heads=n_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable residual scaling

    def forward(self, x, context, context_mask=None):
        key_padding_mask = None if context_mask is None else ~context_mask.bool()
        x_norm = self.norm1(x)
        cross_output, _ = self.cross_attn(x_norm, context, context, key_padding_mask=key_padding_mask)
        return self.norm2(x + self.alpha * self.dropout(cross_output))
    
class GPTWithInjectedCrossAttention(nn.Module):
    def __init__(self, gpt, cross_hidden_dim):
        super().__init__()
        self.transformer_blocks = gpt.transformer.h
        self.cross_blocks = nn.ModuleList([
            NormalizedGatedCrossAttentionBlock(
                dim=gpt.config.n_embd, cross_dim=cross_hidden_dim, n_heads=gpt.config.n_head
            ) for _ in range(len(self.transformer_blocks))
        ])
        self.wte = gpt.transformer.wte
        self.wpe = gpt.transformer.wpe
        self.ln_f = gpt.transformer.ln_f
        self.lm_head = gpt.lm_head

    def forward(self, input_ids, attention_mask, cross_context, cross_mask):
        B, L = input_ids.size()
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0)
        hidden_states = self.wte(input_ids) + self.wpe(pos_ids)

        for i, block in enumerate(self.transformer_blocks):
            hidden_states = block(hidden_states)[0]
            hidden_states = self.cross_blocks[i](hidden_states, cross_context, cross_mask)

        return self.lm_head(self.ln_f(hidden_states))
    
class LearnableFusionBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, text_hidden, dna_hidden):
        combined = torch.cat([text_hidden, dna_hidden], dim=-1)
        gate = self.gate(combined)
        fused = gate * text_hidden + (1 - gate) * dna_hidden
        return self.linear(torch.cat([fused, dna_hidden], dim=-1)) 
    
class ModalityAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.adapter(x)

class ContrastiveTokenHead(nn.Module):
    def __init__(self, hidden_dim, proj_dim=256, dropout=0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, hidden, mask):
        mask = mask.unsqueeze(-1).expand_as(hidden).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.projector(pooled)

class DNATextFusion(nn.Module):
    def __init__(self, dna_model, text_model, cross_attn_interval=4):
        super().__init__()
        self.dna_model = dna_model
        self.text_model = text_model

        dna_hidden = dna_model.config.hidden_size
        text_hidden = text_model.config.n_embd
        self.hidden_dim = text_hidden

        # Modality-specific adapters
        self.dna_adapter = ModalityAdapter(dna_hidden)
        self.text_adapter = ModalityAdapter(text_hidden)

        # Cross-modality projection
        self.dna_to_text_proj = nn.Linear(dna_hidden, text_hidden)
        self.text_to_dna_proj = nn.Linear(text_hidden, dna_hidden)

        # Cross-attention blocks injection at interval
        self.gpt_blocks = text_model.transformer.h
        self.cross_blocks = nn.ModuleList([
            NormalizedGatedCrossAttentionBlock(
                dim=text_hidden, cross_dim=text_hidden, n_heads=text_model.config.n_head
            ) if i % cross_attn_interval == 0 else None 
            for i in range(len(self.gpt_blocks))
        ])

        # Fusion block for deeper integration
        self.fusion_block = LearnableFusionBlock(text_hidden)

        # Unified contrastive projection head
        self.joint_projection = nn.Sequential(
            nn.Linear(text_hidden * 2, text_hidden),
            nn.GELU(),
            nn.LayerNorm(text_hidden)
        )

        # Final LM head
        self.ln_f = text_model.transformer.ln_f
        self.lm_head = text_model.lm_head

    def encode_dna(self, dna_inputs):
        dna_outputs = self.dna_model(
            input_ids=dna_inputs["input_ids"],
            attention_mask=dna_inputs["attention_mask"],
            output_hidden_states=True
        )
        dna_hidden = dna_outputs.hidden_states[-1]
        adapted_dna = self.dna_adapter(dna_hidden)
        return self.dna_to_text_proj(adapted_dna)

    def encode_text(self, text_inputs):
        text_outputs = self.text_model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            output_hidden_states=True
        )
        text_hidden = text_outputs.hidden_states[-1]
        adapted_text = self.text_adapter(text_hidden)
        return adapted_text

    def forward(self, dna_inputs, text_inputs):
        # Modality-specific encoding and adaptation
        dna_context = self.encode_dna(dna_inputs)  # [B, L_dna, H_text]
        text_hidden = self.encode_text(text_inputs)  # [B, L_text, H_text]

        hidden_states = text_hidden

        # Inject gated cross-attention selectively
        for block, cross_block in zip(self.gpt_blocks, self.cross_blocks):
            hidden_states = block(hidden_states)[0]
            if cross_block is not None:
                hidden_states = cross_block(
                    hidden_states, dna_context, dna_inputs["attention_mask"]
                )

        # Final fusion for joint embedding
        fused_representation = self.fusion_block(
            hidden_states.mean(dim=1),  # Text mean pool
            dna_context.mean(dim=1)     # DNA mean pool
        )

        joint_embedding = self.joint_projection(torch.cat([
            hidden_states.mean(dim=1),
            dna_context.mean(dim=1)
        ], dim=-1))

        # LM logits from final hidden states
        logits = self.lm_head(self.ln_f(hidden_states))

        return {
            "logits": logits,
            "dna_logits": logits,  # Add this to match what's expected in training loop
            "joint_embedding": joint_embedding,
            "fused_representation": fused_representation
        }

    @torch.no_grad()
    def generate_text_from_dna(self, dna_inputs, max_length=128, temperature=0.8):
        self.eval()
        dna_context = self.encode_dna(dna_inputs)
        input_ids = torch.full(
            (dna_inputs["input_ids"].size(0), 1),
            self.text_model.config.bos_token_id,
            device=dna_inputs["input_ids"].device
        )

        for _ in range(max_length):
            hidden_states = self.encode_text({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)})

            for block, cross_block in zip(self.gpt_blocks, self.cross_blocks):
                hidden_states = block(hidden_states)[0]
                if cross_block is not None:
                    hidden_states = cross_block(hidden_states, dna_context, dna_inputs["attention_mask"])

            logits = self.lm_head(self.ln_f(hidden_states[:, -1])) / temperature
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if (next_token == self.text_model.config.eos_token_id).all():
                break

        return input_ids

    @torch.no_grad()
    def generate_dna_from_text(self, text_inputs, max_length=512, temperature=0.8):
        """Generate DNA sequences from text inputs."""
        self.eval()
        text_hidden = self.encode_text(text_inputs)
        
        # Start with a single token (typically [CLS] or [BOS])
        input_ids = torch.full(
            (text_inputs["input_ids"].size(0), 1),
            self.dna_model.config.bos_token_id if hasattr(self.dna_model.config, 'bos_token_id') else 0,
            dtype=torch.long,
            device=text_inputs["input_ids"].device
        )
        
        # Project text to DNA space
        text_in_dna_space = self.text_to_dna_proj(text_hidden)
        
        # Generate DNA sequence token by token
        for _ in range(max_length - 1):
            # Create attention mask for current sequence
            attention_mask = torch.ones_like(input_ids)
            
            # Get DNA model outputs for current sequence
            dna_outputs = self.dna_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=text_in_dna_space,  # Condition on text
                encoder_attention_mask=text_inputs["attention_mask"],
                output_hidden_states=True
            )
            
            # Get logits for next token prediction
            next_token_logits = dna_outputs.logits[:, -1, :] / temperature
            
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if we've generated an EOS token
            if (next_token == self.dna_model.config.eos_token_id).all():
                break
            
        return input_ids


model = DNATextFusion(dna_model=dna_model, text_model=text_model).to(device)
freeze_gpt_layers(model, num_layers_to_freeze=FROZEN_LAYERS)

#== optimizer ==
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
steps_per_epoch = len(train_loader)
total_steps = (NUM_EPOCHS * steps_per_epoch) // GRAD_ACCUM_STEPS
warmup_steps = int(0.1 * total_steps)
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

#=======================================================================================================
#                                                                                            Evaluation
#=======================================================================================================

wandb.init(project="dna-text-fusion", name=f"biogpt-dna-fusion")

def log_wandb_step(loss_dict, step, epoch, eta_min=None):
    log_data = {**loss_dict, "step": step, "epoch": epoch}
    if eta_min is not None:
        log_data["eta_min"] = eta_min
    wandb.log(log_data)

def evaluate(model, dataloader, tokenizer, device, show_examples=5, current_epoch=None):
    """
    Evaluate the model on DNA→Text generation with metrics and example logging to wandb
    """
    model.eval()
    all_text_preds, all_text_refs = [], []  # For DNA→Text
    dna_from_text_preds, dna_from_text_refs = [], []  # For Text→DNA
    example_pairs_dna_to_text = []  # DNA→Text examples
    example_pairs_text_to_dna = []  # Text→DNA examples

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Prepare inputs
            dna_inputs = {
                "input_ids": batch["dna_input_ids"].to(device),
                "attention_mask": batch["dna_attention_mask"].to(device),
            }
            text_inputs = {
                "input_ids": batch["text_input_ids"].to(device),
                "attention_mask": batch["text_attention_mask"].to(device),
            }
            
            # 1. Evaluate DNA→Text generation
            generated_text_ids = model.generate_text_from_dna(dna_inputs, max_length=tokenizer.model_max_length)
            generated_texts = tokenizer.batch_decode(generated_text_ids, skip_special_tokens=True)
            reference_texts = tokenizer.batch_decode(batch["text_input_ids"], skip_special_tokens=True)
            
            all_text_preds.extend(generated_texts)
            all_text_refs.extend(reference_texts)
            example_pairs_dna_to_text.extend(list(zip(reference_texts, generated_texts)))
            
            # 2. Evaluate Text→DNA generation 
            # Note: This assumes you have a generate_dna_from_text method in your model
            if hasattr(model, 'generate_dna_from_text'):
                generated_dna_ids = model.generate_dna_from_text(text_inputs, max_length=512)
                generated_dnas = dna_tokenizer.batch_decode(generated_dna_ids, skip_special_tokens=True)
                reference_dnas = dna_tokenizer.batch_decode(batch["dna_input_ids"], skip_special_tokens=True)
                
                dna_from_text_preds.extend(generated_dnas)
                dna_from_text_refs.extend(reference_dnas)
                example_pairs_text_to_dna.extend(list(zip(reference_dnas, generated_dnas)))
            
            # Limit the number of batches processed for faster evaluation
            if batch_idx >= 5:  # Process only first 5 batches
                break

    # Create and log tables with examples
    if example_pairs_dna_to_text:
        dna_to_text_table = wandb.Table(columns=["Reference Text", "Generated Text"])
        for ref, pred in random.sample(example_pairs_dna_to_text, k=min(show_examples, len(example_pairs_dna_to_text))):
            dna_to_text_table.add_data(ref, pred)
        wandb.log({f"dna_to_text_examples/epoch_{current_epoch}": dna_to_text_table})
    
    if example_pairs_text_to_dna:
        text_to_dna_table = wandb.Table(columns=["Reference DNA", "Generated DNA"])
        for ref, pred in random.sample(example_pairs_text_to_dna, k=min(show_examples, len(example_pairs_text_to_dna))):
            text_to_dna_table.add_data(ref[:100]+"...", pred[:100]+"...")  # Truncate long DNA sequences
        wandb.log({f"text_to_dna_examples/epoch_{current_epoch}": text_to_dna_table})

    # Calculate metrics for DNA→Text
    metrics = {}
    
    # BLEU score
    smoothie = SmoothingFunction().method4
    bleu_scores = []
    for ref, pred in zip(all_text_refs, all_text_preds):
        try:
            score = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
            bleu_scores.append(score)
        except Exception as e:
            print(f"BLEU calculation error: {e}")
    
    if bleu_scores:
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        metrics["bleu_score"] = avg_bleu
    
    # ROUGE scores
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, pred) for ref, pred in zip(all_text_refs, all_text_preds)]
        
        avg_rouge1 = sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores)
        avg_rouge2 = sum(score['rouge2'].fmeasure for score in rouge_scores) / len(rouge_scores)
        avg_rougeL = sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores)
        
        metrics["rouge1"] = avg_rouge1
        metrics["rouge2"] = avg_rouge2
        metrics["rougeL"] = avg_rougeL
    except Exception as e:
        print(f"ROUGE calculation error: {e}")
    
    # Log metrics with epoch prefix
    epoch_metrics = {f"{k}/epoch_{current_epoch}": v for k, v in metrics.items()}
    wandb.log(epoch_metrics)
    
    # Also log without prefix to get nice curves
    wandb.log(metrics)
    
    # Print metrics summary
    print(f"🧠 Evaluation Scores - BLEU: {metrics.get('bleu_score', 0):.4f}, "
          f"ROUGE-1: {metrics.get('rouge1', 0):.4f}, ROUGE-L: {metrics.get('rougeL', 0):.4f}")
    
    return metrics.get("bleu_score", 0)  # Return BLEU score for early stopping


#=======================================================================================================
#                                                                                         Training Loop
#=======================================================================================================

def mean_pool(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = (hidden_states * mask).sum(dim=1)
    summed_mask = mask.sum(dim=1).clamp(min=1e-9)
    return summed / summed_mask

ce_loss_fn = CrossEntropyLoss()
cosine_loss_fn = CosineEmbeddingLoss()
epochs_no_improve = 0
scaler = GradScaler()

def info_nce_loss(features, temperature=0.07, batch_size=None):
    """
    InfoNCE contrastive loss for better representation learning.
    
    Args:
        features: Tensor of shape [2*batch_size, feature_dim] containing the features of the positive pairs.
            The first batch_size elements are anchors, and the second batch_size elements are positives.
        temperature: Temperature parameter to scale the similarity scores.
        batch_size: Size of the batch. If None, assumes features has shape [2*B, D] and infers batch_size=B.
    
    Returns:
        InfoNCE loss
    """
    if batch_size is None:
        batch_size = features.shape[0] // 2
        
    # Normalize the features
    features = F.normalize(features, dim=1)
    
    # Create similarity matrix
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # Create labels
    labels = torch.cat([
        torch.arange(batch_size, 2*batch_size),  # Positives for first half (anchors)
        torch.arange(0, batch_size)              # Positives for second half
    ]).to(features.device)
    
    # Create mask for the positive pairs (diagonal blocks)
    mask = torch.zeros_like(similarity_matrix)
    mask.scatter_(1, labels.unsqueeze(1), 1)
    
    # InfoNCE loss calculation
    exp_sim = torch.exp(similarity_matrix)
    log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
    mean_log_prob = (mask * log_prob).sum(1) / mask.sum(1)
    
    return -mean_log_prob.mean()

def dual_softmax_loss(similarity_matrix, temperature=0.07):
    """
    Dual softmax loss for bidirectional similarity optimization.
    Helps with cross-modal retrieval by optimizing in both directions.
    
    Args:
        similarity_matrix: Similarity matrix of shape [batch_size, batch_size]
        temperature: Temperature parameter
    
    Returns:
        Dual softmax loss
    """
    sim_row_scaled = similarity_matrix / temperature
    sim_col_scaled = similarity_matrix.t() / temperature
    
    # Row-wise softmax (text-to-dna retrieval)
    row_labels = torch.arange(sim_row_scaled.size(0)).to(sim_row_scaled.device)
    row_loss = F.cross_entropy(sim_row_scaled, row_labels)
    
    # Column-wise softmax (dna-to-text retrieval)
    col_labels = torch.arange(sim_col_scaled.size(0)).to(sim_col_scaled.device)
    col_loss = F.cross_entropy(sim_col_scaled, col_labels)
    
    # Average the losses
    return (row_loss + col_loss) / 2.0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    grad_norms = []

    optimizer.zero_grad()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    epoch_start_time = time.time()
    
    for step, batch in enumerate(progress_bar):
        dna_inputs = {k: v.to(device) for k, v in batch.items() if "dna" in k}
        text_inputs = {k: v.to(device) for k, v in batch.items() if "text" in k}

        # 🔥 🔒 Safety first: catch any rogue tokens before CUDA meltdown
        assert dna_inputs["input_ids"].max().item() < dna_model.config.vocab_size, \
            f"OOV token in dna_inputs: {dna_inputs['input_ids'].max().item()}"
        assert text_inputs["input_ids"].max().item() < text_model.config.vocab_size, \
            f"OOV token in text_inputs: {text_inputs['input_ids'].max().item()}"

        # Then move to device
        dna_inputs = {k: v.to(device) for k, v in dna_inputs.items()}
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}


        # Text → DNA prediction (exact match)
        dna_labels = batch["dna_input_ids"].to(device)  # shape: (B, L)
        outputs = model(dna_inputs=dna_inputs, text_inputs=text_inputs)  # Forward pass
        dna_logits = outputs["dna_logits"]  # Extract DNA logits

        # Cross-entropy loss for Text → DNA (exact token match)
        text2dna_loss = ce_loss_fn(
            dna_logits.view(-1, dna_logits.size(-1)),
            dna_labels.view(-1)
        )

        # DNA → Text prediction (cross-entropy + contrastive loss)
        text_preds = model.generate_text_from_dna(dna_inputs)  # shape: (B, L)
        text_labels = batch["text_input_ids"].to(device)  # ground truth text
        
        # Cross-entropy loss for DNA → Text
        # Get logits for the generated text
        with autocast():
            dna2text_outputs = model(dna_inputs, {"input_ids": text_preds[:, :-1], "attention_mask": torch.ones_like(text_preds[:, :-1])})
            dna2text_logits = dna2text_outputs["logits"]
            dna2text_ce_loss = ce_loss_fn(
                dna2text_logits.reshape(-1, dna2text_logits.size(-1)),
                text_labels[:, 1:text_preds.size(1)].reshape(-1)
            )
        
        # Convert token IDs to strings (detokenize) for contrastive loss
        predicted_sentences = text_tokenizer.batch_decode(text_preds, skip_special_tokens=True)
        target_sentences = text_tokenizer.batch_decode(batch["text_input_ids"], skip_special_tokens=True)

        # Cosine similarity loss for DNA → Text (semantic similarity)
        with torch.no_grad():
            # Get embeddings directly from text_model
            text_model.eval()
            
            # Get hidden states for predicted and target sentences
            pred_encodings = text_tokenizer(
                predicted_sentences, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=MAX_SEQ_LENGTH
            ).to(device)
            
            target_encodings = text_tokenizer(
                target_sentences, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=MAX_SEQ_LENGTH
            ).to(device)
            
            pred_outputs = text_model(
                input_ids=pred_encodings.input_ids,
                attention_mask=pred_encodings.attention_mask,
                output_hidden_states=True
            )
            
            target_outputs = text_model(
                input_ids=target_encodings.input_ids,
                attention_mask=target_encodings.attention_mask,
                output_hidden_states=True
            )
            
            # Get embeddings from hidden states
            pred_embeddings = mean_pool(
                pred_outputs.hidden_states[-1],
                pred_encodings.attention_mask
            )
            target_embeddings = mean_pool(
                target_outputs.hidden_states[-1],
                target_encodings.attention_mask
            )
            
        # Calculate contrastive loss with temperature annealing
        cosine_targets = torch.ones(pred_embeddings.size(0)).to(device)  # All pairs should be similar
        temp = max(0.07, 0.5 - (epoch / NUM_EPOCHS) * 0.4)  # Temperature annealing
        dna2text_contrastive_loss = cosine_loss_fn(pred_embeddings, target_embeddings, cosine_targets) / temp
        
        # Dynamic weighting for different loss components
        global_step = step + epoch * len(train_loader)
        total_steps = NUM_EPOCHS * len(train_loader)
        alpha = SEMANTIC_LOSS_WEIGHT * max(0.1, min(1.0, global_step / (0.3 * total_steps)))
        
        # Combine all losses with dynamic weights
        dna2text_loss = alpha * dna2text_contrastive_loss + (1 - alpha) * dna2text_ce_loss
        loss = text2dna_loss + dna2text_loss

        # InfoNCE loss
        features = torch.cat([pred_embeddings, target_embeddings], dim=0)
        info_nce = info_nce_loss(features, temperature=temp)

        # Dual softmax loss
        similarity_matrix = torch.matmul(pred_embeddings, target_embeddings.T)
        dual_softmax = dual_softmax_loss(similarity_matrix, temperature=temp)

        # Combine all losses
        loss = loss + info_nce + dual_softmax
        loss = loss / GRAD_ACCUM_STEPS

        # Track gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norms.append(grad_norm.item())

        # 🔁 wandb step log
        current_lr = optimizer.param_groups[0]['lr']
        # Calculate remaining training time (ETA)
        steps_done = step + 1 + epoch * len(train_loader)
        total_training_steps = NUM_EPOCHS * len(train_loader)
        if step > 0:
            steps_per_sec = step / ((time.time() - epoch_start_time) or 1)
            remaining_steps = total_training_steps - steps_done
            eta_seconds = remaining_steps / steps_per_sec
            eta_minutes = eta_seconds / 60
        else:
            eta_minutes = None
            epoch_start_time = time.time()
            
        wandb.log({
            "step_loss": loss.item(),
            "text2dna_loss": text2dna_loss.item(),
            "dna2text_loss": dna2text_loss.item(),
            "dna2text_ce_loss": dna2text_ce_loss.item(),
            "dna2text_contrastive_loss": dna2text_contrastive_loss.item(),
            "info_nce_loss": info_nce.item(),
            "dual_softmax_loss": dual_softmax.item(),
            "contrastive_temperature": temp,
            "alpha_weight": alpha,
            "grad_norm": grad_norm.item(),
            "learning_rate": current_lr,
            "step": steps_done,
            "eta_min": eta_minutes
        })

        # Update progress bar with current loss
        progress_bar.set_postfix({
            "loss": loss.item(),
            "T→D": text2dna_loss.item(),
            "D→T": dna2text_loss.item(),
            "temp": temp,
            "α": alpha
        })

        with autocast():
            outputs = model(dna_inputs, text_inputs)
            loss = text2dna_loss + SEMANTIC_LOSS_WEIGHT * dna2text_loss

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    avg_grad_norm = sum(grad_norms) / len(grad_norms)

    # Evaluate
    val_bleu = evaluate(model, test_loader, text_tokenizer, device, current_epoch=epoch+1)

    print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f} | GradNorm: {avg_grad_norm:.4f} | Val BLEU: {val_bleu:.4f}")

    # Early stopping logic
    if val_bleu > best_val_f1:
        best_val_f1 = val_bleu
        epochs_no_improve = 0
        print("✨ New best model found!")
        torch.save(model.state_dict(), "best_model.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print("⏹️ Early stopping triggered.")
            break