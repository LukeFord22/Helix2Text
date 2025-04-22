import torch
import pandas as pd
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM, BitsAndBytesConfig
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from transformers import GPT2Config

#=======================================================================================================
#                                                                                         Model loading
#=======================================================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    quantization_method="llm_int8",  # Optional: Specify quantization method
)

# Load METAGENE-1-BnB-4Bit model with quantization
dna_model_path = "metagene-ai/METAGENE-1-BnB-4Bit"
dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_path, trust_remote_code=True)
dna_model = AutoModelForCausalLM.from_pretrained(
    dna_model_path,
    trust_remote_code=True,
    quantization_config=bnb_config,  # Use the new quantization configuration
    device_map="auto"  # Automatically map layers to GPU if available
).to(device)


#== Distill GPT2 Model ==
text_model_path = "distilgpt2"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_path, trust_remote_code=True)

# Import and use GPT2Config to enable cross-attention
text_config = GPT2Config.from_pretrained(text_model_path)
text_config.add_cross_attention = True  # Enable cross-attention

# Load the model with the modified config
text_model = GPT2LMHeadModel.from_pretrained(
    text_model_path, 
    config=text_config,
    trust_remote_code=True
).to(device)

if text_tokenizer.pad_token is None:
    text_tokenizer.pad_token = text_tokenizer.eos_token

#=======================================================================================================
#                                                                                   Adjustable Parameters
#=======================================================================================================
torch.autograd.set_detect_anomaly(True)

# Add early stopping parameters
EARLY_STOP_PATIENCE = 3
best_val_f1 = 0.3
epochs_no_improve = 3

#Run parameters
BATCH_SIZE = 2
NUM_EPOCHS = 8
LEARNING_RATE = 3e-5
MAX_SEQ_LENGTH = 512
GRAD_ACCUM_STEPS = 5
TRAIN_DATA_FRACTION = 1
SEMANTIC_LOSS_WEIGHT = 0.5

#=======================================================================================================
#                                                                                        Dataset Loading
#======================================================================================================= 

df = pd.read_csv("ncbi_search.csv")

sampled_df = df.sample(frac=TRAIN_DATA_FRACTION, random_state=42)
train_df = sampled_df.sample(frac=0.9, random_state=42)
test_df = sampled_df.drop(train_df.index)

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
})

def tokenize_fn(example):
    dna_enc = dna_tokenizer(
        example["sequence"], 
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
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Fusion Model
class DNATextFusion(nn.Module):

    def get_decoder_hidden_states(self, model, input_ids, attention_mask):
        device = next(model.parameters()).device  # Dynamically grab model’s actual device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False
        )
        return outputs.hidden_states[-1]
    
    def generate_text_from_dna(self, dna_inputs, max_length=512, num_beams=4):
        """
        Generate text from DNA inputs using the fusion model's text generation capabilities.
        
        Args:
            dna_inputs: Dictionary with DNA input_ids and attention_mask
            max_length: Maximum length of generated text
            num_beams: Number of beams for beam search
            
        Returns:
            Generated text token IDs
        """
        # Get encoder-style hidden states from DNA model
        dna_hidden = self.get_decoder_hidden_states(
            self.dna_model,
            dna_inputs["input_ids"],
            dna_inputs["attention_mask"]
        )
        
        # Project DNA hidden states to text space
        projected_dna = self.dna_to_text_proj(dna_hidden.to(self.dna_to_text_proj.weight.dtype))
        
        # Generate text using the text model with projected DNA as conditioning
        generated_ids = self.text_model.generate(
            input_ids=dna_inputs["input_ids"][:, :1],  # Just use first token or BOS token
            encoder_hidden_states=projected_dna,
            encoder_attention_mask=dna_inputs["attention_mask"],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
        
        return generated_ids

    def __init__(self, dna_model, text_model):
        super(DNATextFusion, self).__init__()
        
        # DNA and Text Models
        self.dna_model = dna_model
        self.text_model = text_model

        # Dynamically retrieve the hidden sizes
        dna_hidden_size = dna_model.config.hidden_size
        text_hidden_size = text_model.config.n_embd

        # Separate projection layers for DNA and Text hidden states
        self.dna_to_text_proj = nn.Linear(dna_hidden_size, text_hidden_size)
        self.text_to_dna_proj = nn.Linear(text_hidden_size, dna_hidden_size)

    def forward(self, dna_inputs, text_inputs, attention_mask=None):
        # Get encoder-style hidden states from decoder-only models
        dna_hidden = self.get_decoder_hidden_states(
            self.dna_model,
            dna_inputs["input_ids"],
            dna_inputs["attention_mask"]
        )

        text_hidden = self.get_decoder_hidden_states(
            self.text_model,
            text_inputs["input_ids"],
            text_inputs["attention_mask"]
        )

        # Use separate projection layers for compatibility
        projected_dna = self.dna_to_text_proj(dna_hidden.to(self.dna_to_text_proj.weight.dtype))
        projected_text = self.text_to_dna_proj(text_hidden.to(self.text_to_dna_proj.weight.dtype))

        # Print vocabulary information for debugging
        print("DNA Input max ID:", dna_inputs["input_ids"].max().item())
        print("DNA Vocab size:", self.dna_model.config.vocab_size)
        print("Text Input max ID:", text_inputs["input_ids"].max().item())
        print("Text Vocab size:", self.text_model.config.vocab_size)

        # Create safe input IDs for cross-model use
        # Clamp values to be within each model's vocabulary range
        safe_dna_ids = torch.clamp(dna_inputs["input_ids"], 0, self.dna_model.config.vocab_size - 1)
        safe_text_ids = torch.clamp(text_inputs["input_ids"], 0, self.text_model.config.vocab_size - 1)
        
        # DNA-from-Text decoding - use DNA's own token IDs for input_ids
        dna_decoder_outputs = self.dna_model(
            input_ids=safe_dna_ids,  # Use DNA's own safe IDs
            encoder_hidden_states=projected_text,
            encoder_attention_mask=text_inputs["attention_mask"],
            attention_mask=attention_mask,
        )

        # Text-from-DNA decoding - use Text's own token IDs for input_ids
        text_decoder_outputs = self.text_model(
            input_ids=safe_text_ids,  # Use Text's own safe IDs
            encoder_hidden_states=projected_dna,
            encoder_attention_mask=dna_inputs["attention_mask"],
            attention_mask=attention_mask,
        )

        return {
            "dna_logits": dna_decoder_outputs.logits,
            "text_logits": text_decoder_outputs.logits
        }


model = DNATextFusion(dna_model=dna_model, text_model=text_model).to(device)

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

def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            dna_inputs = {
                "input_ids": batch["dna_input_ids"].to(device),
                "attention_mask": batch["dna_attention_mask"].to(device),
            }
            text_inputs = {
                "input_ids": batch["text_input_ids"].to(device),
                "attention_mask": batch["text_attention_mask"].to(device),
            }
            labels = batch["label"].to(device)

            outputs = model(dna_inputs, text_inputs)
            preds = outputs.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return f1_score(all_labels, all_preds, average="macro")


#=======================================================================================================
#                                                                                         Training Loop
#=======================================================================================================

ce_loss_fn = CrossEntropyLoss()
cosine_loss_fn = CosineEmbeddingLoss()

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    grad_norms = []

    optimizer.zero_grad()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for step, batch in enumerate(progress_bar):
        dna_inputs = {
            "input_ids": batch["dna_input_ids"].to(device),
            "attention_mask": batch["dna_attention_mask"].to(device),
        }
        text_inputs = {
            "input_ids": batch["text_input_ids"].to(device),
            "attention_mask": batch["text_attention_mask"].to(device),
        }

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

        # DNA → Text prediction (semantic match)
        text_preds = model.generate_text_from_dna(dna_inputs)  # shape: (B, L)
        # Convert token IDs to strings (detokenize)
        predicted_sentences = text_tokenizer.batch_decode(text_preds, skip_special_tokens=True)
        target_sentences = text_tokenizer.batch_decode(batch["text_input_ids"], skip_special_tokens=True)

        # Cosine similarity loss for DNA → Text (semantic similarity)
        with torch.no_grad():
            # Get embeddings directly from text_model
            text_model.eval()
            
            # Get hidden states for predicted and target sentences
            pred_outputs = text_model(input_ids=text_tokenizer(predicted_sentences, return_tensors="pt").input_ids.to(device), attention_mask=text_tokenizer(predicted_sentences, return_tensors="pt").attention_mask.to(device))
            target_outputs = text_model(input_ids=text_tokenizer(target_sentences, return_tensors="pt").input_ids.to(device), attention_mask=text_tokenizer(target_sentences, return_tensors="pt").attention_mask.to(device))
            
            # Assuming we take the last hidden state for embeddings (you could also try other pooling strategies)
            pred_embeddings = pred_outputs.last_hidden_state[:, 0, :]  # Taking the embedding of the [CLS] token (you can modify this depending on your strategy)
            target_embeddings = target_outputs.last_hidden_state[:, 0, :]  # Taking the embedding of the [CLS] token (you can modify this depending on your strategy)
            
        # Cosine similarity loss
        cosine_targets = torch.ones(pred_embeddings.size(0)).to(device)  # All pairs should be similar
        dna2text_loss = cosine_loss_fn(pred_embeddings, target_embeddings, cosine_targets)

        # Combine losses (tune SEMANTIC_LOSS_WEIGHT as needed)
        loss = text2dna_loss + SEMANTIC_LOSS_WEIGHT * dna2text_loss

        # Track gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norms.append(grad_norm.item())

        # Update progress bar with current loss
        progress_bar.set_postfix({
            "loss": loss.item(),
            "T→D": text2dna_loss.item(),
            "D→T": dna2text_loss.item(),
            "grad_norm": grad_norm.item()
        })

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    avg_grad_norm = sum(grad_norms) / len(grad_norms)

    # Evaluate
    val_f1 = evaluate(model, test_loader)

    print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f} | GradNorm: {avg_grad_norm:.4f} | Val F1: {val_f1:.4f}")

    # Early stopping logic
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_no_improve = 0
        print("✨ New best model found!")
        torch.save(model.state_dict(), "best_model.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print("⏹️ Early stopping triggered.")
            break