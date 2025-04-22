import torch
import pandas as pd
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel, BertConfig, GPT2LMHeadModel
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss
from tqdm import tqdm


#=======================================================================================================
#                                                                                         Model loading
#=======================================================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#== DNABERT-2 Model ==
dna_model_path = "zhihan1996/DNABERT-2-117M"
dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_path, trust_remote_code=True)
config = BertConfig.from_pretrained(dna_model_path)
dna_model = AutoModel.from_pretrained(dna_model_path, config=config, trust_remote_code=True).to(device)


#== Distill GPT2 Model ==
text_model_path = "distilgpt2"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_path, trust_remote_code=True)
text_model = GPT2LMHeadModel.from_pretrained(text_model_path, trust_remote_code=True).to(device)

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
BATCH_SIZE = 4
NUM_EPOCHS = 8
LEARNING_RATE = 3e-5
MAX_SEQ_LENGTH = 512
GRAD_ACCUM_STEPS = 4
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

# Fusion Model
class DNATextFusion(nn.Module):
    def __init__(self, dna_model, text_model, hidden_size=768):
        super(DNATextFusion, self).__init__()
        
        # Shared Encoder (e.g., T5, BERT)
        self.encoder = dna_model.encoder  # Shared encoder
        
        # DNA and Text Decoders
        self.dna_decoder = dna_model.decoder  # Decoder for DNA generation
        self.text_decoder = text_model.decoder  # Decoder for Text generation

        # Project encoder outputs to match the decoder's input size
        self.encoder_to_decoder_proj = nn.Linear(hidden_size, self.dna_decoder.config.d_model)

    def forward(self, dna_inputs, text_inputs, attention_mask=None):
        # Encoder (shared for both DNA and Text)
        # Process DNA sequence through the encoder
        dna_outputs = self.encoder(input_ids=dna_inputs["input_ids"], attention_mask=dna_inputs["attention_mask"])
        dna_hidden_states = dna_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Process Text sequence through the encoder
        text_outputs = self.encoder(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])
        text_hidden_states = text_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Project encoder hidden states to match the decoder's hidden size (if needed)
        projected_dna_hidden = self.encoder_to_decoder_proj(dna_hidden_states)
        projected_text_hidden = self.encoder_to_decoder_proj(text_hidden_states)

        # Decoder for DNA from Text (Text to DNA generation)
        # Cross-attention on DNA decoder
        dna_decoder_outputs = self.dna_decoder(
            input_ids=text_inputs["input_ids"],  # Text input is used here
            encoder_hidden_states=projected_text_hidden,
            encoder_attention_mask=text_inputs["attention_mask"],
            attention_mask=attention_mask
        )

        # Decoder for Text from DNA (DNA to Text generation)
        # Cross-attention on Text decoder
        text_decoder_outputs = self.text_decoder(
            input_ids=dna_inputs["input_ids"],  # DNA input is used here
            encoder_hidden_states=projected_dna_hidden,
            encoder_attention_mask=dna_inputs["attention_mask"],
            attention_mask=attention_mask
        )

        # Output the logits for both DNA and Text
        return {
            "dna_logits": dna_decoder_outputs.logits,  # Logits for DNA prediction
            "text_logits": text_decoder_outputs.logits   # Logits for Text prediction
        }

    def generate_dna_from_text(self, text_inputs, attention_mask=None):
        # Generate DNA from Text
        text_outputs = self.text_decoder.generate(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            max_length=128  # Define max length for DNA generation
        )
        return text_outputs

    def generate_text_from_dna(self, dna_inputs, attention_mask=None):
        # Generate Text from DNA
        dna_outputs = self.dna_decoder.generate(
            input_ids=dna_inputs["input_ids"],
            attention_mask=dna_inputs["attention_mask"],
            max_length=128  # Define max length for Text generation
        )
        return dna_outputs


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

        # Text → DNA prediction (exact match)
        dna_labels = batch["dna_labels"].to(device)  # shape: (B, L)
        dna_logits = model.generate_dna_from_text(text_inputs)  # shape: (B, L, V)

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