import os
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from time import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel, BertConfig, AutoModelForCausalLM
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import random
import wandb

#=======================================================================================================
#                                                                                  Adjustable Parameters
#=======================================================================================================
torch.autograd.set_detect_anomaly(True)

# Add early stopping parameters
EARLY_STOP_PATIENCE = 5
BEST_VALUE = 0

#Run parameters
BATCH_SIZE = 2
NUM_EPOCHS = 3
LEARNING_RATE = 3e-5
MAX_SEQ_LENGTH = 16
GRAD_ACCUM_STEPS = 4
TRAIN_DATA_FRACTION = 0.01
SEMANTIC_LOSS_WEIGHT = 0.5
FROZEN_LAYERS = 12

#=======================================================================================================
#                                                                                         Model loading
#=======================================================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#== DNABERT-2 Model ==
dna_model_path = "zhihan1996/DNABERT-2-117M"
dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_path, trust_remote_code=True)
config = BertConfig.from_pretrained(dna_model_path)
dna_model = AutoModel.from_pretrained(dna_model_path, config=config, trust_remote_code=True).to(device)


#== BioGPT Model ==
text_model_path = "microsoft/biogpt-large-pubmedqa"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_path, trust_remote_code=True)
if text_tokenizer.pad_token is None:
    text_tokenizer.pad_token = text_tokenizer.eos_token
    text_tokenizer.pad_token_id = text_tokenizer.eos_token_id
text_tokenizer.padding_side = 'left'
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

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "protein_to_dna_sequence_descriptions.csv"))

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
    
    # For training, we need the DNA tokens as both input and labels
    # Copy the input IDs as labels for predicting DNA sequence
    dna_labels = dna_enc["input_ids"].copy()
    
    return {
        "dna_input_ids": dna_enc["input_ids"],
        "dna_attention_mask": dna_enc["attention_mask"],
        "text_input_ids": text_enc["input_ids"],
        "text_attention_mask": text_enc["attention_mask"],
        "dna_labels": dna_labels, 
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

# Fusion Model
class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size, cross_hidden_size, n_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            kdim=cross_hidden_size,
            vdim=cross_hidden_size,
            num_heads=n_heads,
            batch_first=True
        )
        self.gate = nn.Parameter(torch.ones(1))  
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, context, context_mask=None):
        key_padding_mask = None
        if context_mask is not None:
            key_padding_mask = ~context_mask.bool()

        attn_output, _ = self.cross_attn(x, context, context, key_padding_mask=key_padding_mask)
        gated_output = self.gate * attn_output
        return self.norm(x + self.dropout(gated_output))


class GPTWithCrossAttention(nn.Module):
    def __init__(self, gpt_model, cross_hidden_size, insert_every_n=4):
        super().__init__()
        self.gpt = gpt_model
        
        # BioGPT has a different structure - layers are directly in the biogpt encoder
        self.transformer_blocks = gpt_model.biogpt.layers
        
        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(
                hidden_size=gpt_model.config.hidden_size,
                cross_hidden_size=cross_hidden_size,
                n_heads=gpt_model.config.num_attention_heads
            ) for _ in range(len(self.transformer_blocks))
        ])
        self.insert_every_n = insert_every_n
        
        # Get the appropriate embedding layers from BioGPT
        self.wte = gpt_model.biogpt.embed_tokens
        
        # BioGPT uses position embeddings differently
        self.has_position_embedding = hasattr(gpt_model.biogpt, 'embed_positions')
        if self.has_position_embedding:
            self.wpe = gpt_model.biogpt.embed_positions
            
        # Final layer norm and output projection for BioGPT
        self.ln_f = gpt_model.biogpt.layer_norm
        
        # BioGPT doesn't have a separate lm_head attribute, its output layer is integrated
        # We'll create a projection matching the vocabulary size
        self.lm_head = nn.Linear(gpt_model.config.hidden_size, gpt_model.config.vocab_size)
        # Initialize with the same weights as in BioGPT's output projection
        if hasattr(gpt_model, 'output_projection'):
            self.lm_head.weight.data = gpt_model.output_projection.weight.data.clone()
        elif hasattr(gpt_model.biogpt, 'output_projection'): 
            self.lm_head.weight.data = gpt_model.biogpt.output_projection.weight.data.clone()

    def forward(self, input_ids, attention_mask, dna_context, dna_mask=None):
        B, L = input_ids.shape
        
        # Handle embeddings for BioGPT
        hidden_states = self.wte(input_ids)
        
        # Add position embeddings if available
        if self.has_position_embedding:
            position_embeddings = self.wpe(input_ids)
            hidden_states = hidden_states + position_embeddings
        
        # Process through layers with cross-attention
        for i, block in enumerate(self.transformer_blocks):
            # BioGPT expects a different format for attention mask
            # Create an appropriate attention mask format for BioGPT
            attention_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)
            
            # Call the layer - BioGPT takes different parameters
            layer_outputs = block(
                hidden_states,
                attention_mask=attention_mask_expanded
            )
            hidden_states = layer_outputs[0]
            
            # Apply cross-attention
            if i % self.insert_every_n == 0:
                hidden_states = self.cross_attention_blocks[i](hidden_states, dna_context, dna_mask)

        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        return logits, hidden_states

class DNATextFusion(nn.Module):
    def __init__(self, dna_encoder, gpt_model, hidden_size=768):
        super().__init__()
        self.dna_encoder = dna_encoder
        self.dna_cls = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.gpt_cross = GPTWithCrossAttention(
            gpt_model=gpt_model,
            cross_hidden_size=hidden_size
        )
        self.dna_proj = nn.Sequential(
            nn.Linear(hidden_size, gpt_model.config.hidden_size),  # Use hidden_size, not n_embd
            nn.LayerNorm(gpt_model.config.hidden_size)
        )

        # Use text_tokenizer's special tokens since model.config might not have them
        self.bos_token_id = text_tokenizer.bos_token_id
        self.eos_token_id = text_tokenizer.eos_token_id

    def encode_dna(self, dna_inputs):
        dna_out = self.dna_encoder(
            input_ids=dna_inputs["input_ids"],
            attention_mask=dna_inputs["attention_mask"]
        )
        dna_context = dna_out[0] if isinstance(dna_out, tuple) else dna_out.last_hidden_state
        cls_token = self.dna_cls.expand(dna_context.size(0), -1, -1)
        dna_context = torch.cat([cls_token, dna_context], dim=1)
        return self.dna_proj(dna_context)

    def forward(self, dna_inputs, text_inputs):
        dna_context = self.encode_dna(dna_inputs)
        logits, text_hidden = self.gpt_cross(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            dna_context=dna_context,
            dna_mask=None  # if used: adjust mask to account for CLS token
        )
        return {"text_logits": logits, "text_hidden": text_hidden}

    def generate_text_from_dna(self, dna_inputs, max_length=128, temperature=0.7):
        with torch.no_grad():
            dna_context = self.encode_dna(dna_inputs)
            input_ids = torch.full(
                (dna_inputs["input_ids"].size(0), 1),
                text_tokenizer.bos_token_id if text_tokenizer.bos_token_id else text_tokenizer.eos_token_id,
                dtype=torch.long,
                device=dna_inputs["input_ids"].device
            )
            
            # Use temperature sampling instead of argmax
            for _ in range(max_length):
                logits, _ = self.gpt_cross(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    dna_context=dna_context,
                    dna_mask=None
                )
                # Apply temperature and sample
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Optional: Stop if EOS token is generated
                if (next_token == text_tokenizer.eos_token_id).all():
                    break
                    
            return input_ids



model = DNATextFusion(dna_encoder=dna_model, gpt_model=text_model).to(device)
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

def evaluate(model, dataloader, tokenizer, device, show_examples=5, current_epoch=None):
    model.eval()
    all_preds, all_refs = [], []
    example_pairs = []  # Collect examples here first

    with torch.no_grad():
        for batch in dataloader:
            dna_inputs = {
                "input_ids": batch["dna_input_ids"].to(device),
                "attention_mask": batch["dna_attention_mask"].to(device),
            }
            true_texts = batch["text_input_ids"]

            # Generate
            generated_ids = model.generate_text_from_dna(dna_inputs, max_length=tokenizer.model_max_length)
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            reference_texts = tokenizer.batch_decode(true_texts, skip_special_tokens=True)

            all_preds.extend(generated_texts)
            all_refs.extend(reference_texts)
            
            # Just store examples for now, don't log yet
            example_pairs.extend(list(zip(reference_texts, generated_texts)))
            
            # Stop collecting if we have enough examples
            if len(example_pairs) >= show_examples * 2:
                break

    # Create one table with a random sample of examples after all evaluation is done
    sample_table = wandb.Table(columns=["Reference", "Prediction"])
    
    # Take a random sample of the collected examples
    for ref, pred in random.sample(example_pairs, k=min(show_examples, len(example_pairs))):
        sample_table.add_data(ref, pred)
        
    # Log just once with a unique name per epoch
    epoch_name = f"epoch_{current_epoch}" if current_epoch is not None else f"epoch_unknown"
    wandb.log({f"eval_examples/{epoch_name}": sample_table})

    # BLEU or other metrics
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoothie = SmoothingFunction().method4
    bleu_scores = [sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
                   for ref, pred in zip(all_refs, all_preds)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred) for ref, pred in zip(all_refs, all_preds)]
    
    avg_rouge1 = sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores)
    avg_rouge2 = sum(score['rouge2'].fmeasure for score in rouge_scores) / len(rouge_scores)
    avg_rougeL = sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores)
    
    wandb.log({
        f"rouge1/epoch_{current_epoch}": avg_rouge1,
        f"rouge2/epoch_{current_epoch}": avg_rouge2,
        f"rougeL/epoch_{current_epoch}": avg_rougeL
    })
    
    print(f"🧠 Evaluation Scores - BLEU: {avg_bleu:.4f}, ROUGE-1: {avg_rouge1:.4f}, ROUGE-L: {avg_rougeL:.4f}")
    return avg_bleu 


#=======================================================================================================
#                                                                                         Training Loop
#=======================================================================================================

wandb.init(
    project="dna-text-fusion",
    name=f"run-epochs-{NUM_EPOCHS}",
    config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "epochs": NUM_EPOCHS,
        "semantic_loss_weight": SEMANTIC_LOSS_WEIGHT,
        "max_seq_length": MAX_SEQ_LENGTH
    }
)

scaler = GradScaler()
def contrastive_nce_loss(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    # Normalize embeddings
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)

    logits = a @ b.T  # [B, B]
    logits /= temperature

    targets = torch.arange(a.size(0), device=a.device)
    return nn.CrossEntropyLoss()(logits, targets)

def mean_pool(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = (hidden_states * mask).sum(dim=1)
    summed_mask = mask.sum(dim=1).clamp(min=1e-9)
    return summed / summed_mask

epochs_no_improve = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss, grad_norms = 0.0, []
    step_times = []
    optimizer.zero_grad()

    start_time = time()
    progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="step")

    for step, batch in enumerate(train_loader):
        step_start = time()

        dna_inputs = {
            "input_ids": batch["dna_input_ids"].to(device),
            "attention_mask": batch["dna_attention_mask"].to(device),
        }
        text_inputs = {
            "input_ids": batch["text_input_ids"].to(device),
            "attention_mask": batch["text_attention_mask"].to(device),
        }

        outputs = model(dna_inputs, text_inputs)
        text_logits = outputs["text_logits"]
        text_hidden = outputs["text_hidden"]

        # Token-level CE loss (autoregressive)
        ce_loss = F.cross_entropy(
            text_logits.view(-1, text_logits.size(-1)),
            text_inputs["input_ids"].view(-1),
            ignore_index=text_tokenizer.pad_token_id
        )

        # Contrastive supervision from generated sequences
        with torch.no_grad():
            model.eval()
            generated_ids = model.generate_text_from_dna(dna_inputs, max_length=MAX_SEQ_LENGTH)
            model.train()

        pred_sentences = text_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        target_sentences = text_tokenizer.batch_decode(text_inputs["input_ids"], skip_special_tokens=True)

        pred_enc = text_tokenizer(pred_sentences, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to(device)
        target_enc = text_tokenizer(target_sentences, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to(device)

        with torch.no_grad():
            target_hidden = text_model(**target_enc, output_hidden_states=True).hidden_states[-1]

        pred_embed = mean_pool(text_hidden, text_inputs["attention_mask"])
        target_embed = mean_pool(target_hidden, target_enc["attention_mask"])

        with autocast():
            temp = max(0.07, 0.5 - (epoch / NUM_EPOCHS) * 0.4)
            contrastive_loss = contrastive_nce_loss(pred_embed, target_embed, temperature=temp)
            global_step = step + epoch * len(train_loader)
            total_steps = NUM_EPOCHS * len(train_loader)
            alpha = SEMANTIC_LOSS_WEIGHT * max(0.1, min(1.0, global_step / (0.3 * total_steps)))
            loss = alpha * contrastive_loss + (1 - alpha) * ce_loss

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norms.append(grad_norm.item())

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        step_end = time()
        step_duration = step_end - step_start
        step_times.append(step_duration)
        avg_step_time = sum(step_times[-10:]) / min(len(step_times), 10)
        eta_minutes = avg_step_time * (len(train_loader) - (step + 1)) / 60

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "CE": f"{ce_loss.item():.4f}",
            "D→T": f"{contrastive_loss.item():.4f}",
            "α": f"{alpha:.2f}",
            "grad_norm": f"{grad_norm.item():.2f}" if 'grad_norm' in locals() else "-",
            "eta_min": f"{eta_minutes:.1f}"
        })
        progress_bar.update(1)
        total_loss += loss.item()

        # 🔁 wandb step log
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "step_loss": loss.item(),
            "ce_loss": ce_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
            "alpha": alpha,
            "grad_norm": grad_norm.item() if 'grad_norm' in locals() else None,
            "learning_rate": current_lr,
            "step": step + 1 + epoch * len(train_loader),
            "eta_min": eta_minutes
        })

    progress_bar.close()
    epoch_time = time() - start_time
    avg_loss = total_loss / len(train_loader)
    avg_grad_norm = sum(grad_norms) / len(grad_norms)

    # Evaluate
    bleu_score = evaluate(model, test_loader, tokenizer=text_tokenizer, device=device, show_examples=3, current_epoch=epoch+1)

    print(f"[Epoch {epoch+1}] ⏱️ Time: {epoch_time:.1f}s | Loss: {avg_loss:.4f} | GradNorm: {avg_grad_norm:.4f} | BLEU: {bleu_score:.4f}")

    wandb.log({
        "epoch": epoch + 1,
        "avg_loss": avg_loss,
        "avg_grad_norm": avg_grad_norm,
        "bleu_score": bleu_score,
        "epoch_time_sec": epoch_time
    })

    if bleu_score > BEST_VALUE:
        BEST_VALUE = bleu_score
        epochs_no_improve = 0
        print("✨ New best model saved!")
        torch.save(model.state_dict(), "best_model.pt")
        wandb.log({"best_model_saved": True})
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print("⏹️ Early stopping triggered.")
            wandb.log({"early_stopped": True})
            break