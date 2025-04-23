"""DNA-Text fusion model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .attention import NormalizedGatedCrossAttentionBlock
from .components import LearnableFusionBlock, ModalityAdapter
from ..data.tokenizers import load_tokenizers

def load_models(dna_model_path, text_model_path, device='cuda'):
    """
    Load pretrained DNA and text models.
    
    Args:
        dna_model_path: Path to the DNA model
        text_model_path: Path to the text model
        device: Device to load models to
        
    Returns:
        Tuple of (dna_tokenizer, text_tokenizer, dna_model, text_model)
    """
    # Load tokenizers
    dna_tokenizer, text_tokenizer = load_tokenizers(dna_model_path, text_model_path)
    
    # Load DNA model with quantization
    dna_model = AutoModelForCausalLM.from_pretrained(
        dna_model_path,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Load text model
    text_model = AutoModelForCausalLM.from_pretrained(
        text_model_path,
        trust_remote_code=True,
        pad_token_id=text_tokenizer.eos_token_id
    ).to(device)
    
    return dna_tokenizer, text_tokenizer, dna_model, text_model

class DNATextFusion(nn.Module):
    """
    DNA-Text fusion model for bidirectional generation.
    
    This model combines DNA and text encoders with cross-attention
    to enable bidirectional generation and retrieval.
    """
    def __init__(self, dna_model, text_model, cross_attn_interval=4):
        super().__init__()
        self.dna_model = dna_model
        self.text_model = text_model
        
        # Get model dimensions
        dna_hidden = dna_model.config.hidden_size
        text_hidden = text_model.config.hidden_size if hasattr(text_model.config, 'hidden_size') else text_model.config.n_embd
        self.hidden_dim = text_hidden
        
        # Modality-specific adapters
        self.dna_adapter = ModalityAdapter(dna_hidden)
        self.text_adapter = ModalityAdapter(text_hidden)
        
        # Cross-modality projections
        self.dna_to_text_proj = nn.Linear(dna_hidden, text_hidden)
        self.text_to_dna_proj = nn.Linear(text_hidden, dna_hidden)
        
        # Get transformer blocks depending on model architecture
        if hasattr(text_model, 'transformer'):
            # GPT-2 style model
            self.gpt_blocks = text_model.transformer.h
        elif hasattr(text_model, 'biogpt'):
            # BioGPT style model
            self.gpt_blocks = text_model.biogpt.layers
        else:
            raise ValueError(f"Unsupported text model architecture: {type(text_model)}")
        
        # Cross-attention blocks
        self.cross_blocks = nn.ModuleList([
            NormalizedGatedCrossAttentionBlock(
                dim=text_hidden, 
                cross_dim=text_hidden, 
                n_heads=8 if not hasattr(text_model.config, 'n_head') else text_model.config.n_head
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
        
        # Final LM head setup based on model type
        if hasattr(text_model, 'transformer'):
            self.ln_f = text_model.transformer.ln_f
            self.lm_head = text_model.lm_head
        elif hasattr(text_model, 'biogpt'):
            self.ln_f = text_model.biogpt.layer_norm
            self.lm_head = text_model.output_projection if hasattr(text_model, 'output_projection') else text_model.biogpt.output_projection
    
    def encode_dna(self, dna_inputs):
        """
        Encode DNA inputs to text-compatible embeddings.
        
        Args:
            dna_inputs: Dictionary with DNA input_ids and attention_mask
            
        Returns:
            Encoded DNA representations
        """
        dna_outputs = self.dna_model(
            input_ids=dna_inputs["input_ids"],
            attention_mask=dna_inputs["attention_mask"],
            output_hidden_states=True
        )
        dna_hidden = dna_outputs.hidden_states[-1]
        adapted_dna = self.dna_adapter(dna_hidden)
        return self.dna_to_text_proj(adapted_dna)
    
    def encode_text(self, text_inputs):
        """
        Encode text inputs to hidden representations.
        
        Args:
            text_inputs: Dictionary with text input_ids and attention_mask
            
        Returns:
            Encoded text representations
        """
        # Handle different model architectures
        if hasattr(self.text_model, 'transformer'):
            # GPT-2 style
            text_outputs = self.text_model.transformer(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                output_hidden_states=True
            )
        else:
            # Other architectures like BioGPT
            text_outputs = self.text_model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                output_hidden_states=True
            )
        
        # Get the last hidden state
        if hasattr(text_outputs, 'hidden_states'):
            text_hidden = text_outputs.hidden_states[-1]
        elif hasattr(text_outputs, 'last_hidden_state'):
            text_hidden = text_outputs.last_hidden_state
        else:
            raise ValueError("Could not extract hidden states from text model outputs")
        
        adapted_text = self.text_adapter(text_hidden)
        return adapted_text
    
    def forward(self, dna_inputs, text_inputs):
        """
        Forward pass for DNA-Text fusion model.
        
        Args:
            dna_inputs: Dictionary with DNA input_ids and attention_mask
            text_inputs: Dictionary with text input_ids and attention_mask
            
        Returns:
            Dictionary containing model outputs
        """
        # Modality-specific encoding and adaptation
        dna_context = self.encode_dna(dna_inputs)  # [B, L_dna, H_text]
        text_hidden = self.encode_text(text_inputs)  # [B, L_text, H_text]
        
        hidden_states = text_hidden
        
        # Inject gated cross-attention selectively
        for block, cross_block in zip(self.gpt_blocks, self.cross_blocks):
            # Apply the model's own self-attention block
            if hasattr(block, '__call__'):
                hidden_states = block(hidden_states)[0]
            else:
                # Handle models with different block structure
                hidden_states = block(hidden_states, attention_mask=None)[0]
                
            # Apply cross-attention if available at this layer
            if cross_block is not None:
                hidden_states = cross_block(
                    hidden_states, dna_context, dna_inputs["attention_mask"]
                )
        
        # Mean pooling for sequence-level representations
        text_pooled = hidden_states.mean(dim=1)  # Text mean pool
        dna_pooled = dna_context.mean(dim=1)     # DNA mean pool
        
        # Final fusion for joint embedding
        fused_representation = self.fusion_block(text_pooled, dna_pooled)
        
        # Joint embedding for contrastive learning
        joint_embedding = self.joint_projection(torch.cat([text_pooled, dna_pooled], dim=-1))
        
        # LM logits from final hidden states
        logits = self.lm_head(self.ln_f(hidden_states))
        
        return {
            "logits": logits,
            "dna_logits": logits,  # Compatible with existing training loop
            "joint_embedding": joint_embedding,
            "fused_representation": fused_representation,
            "text_hidden": hidden_states,
            "dna_hidden": dna_context
        }
    
    @torch.no_grad()
    def generate_text_from_dna(self, dna_inputs, max_length=128, temperature=0.8):
        """
        Generate text from DNA inputs.
        
        Args:
            dna_inputs: Dictionary with DNA input_ids and attention_mask
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text token IDs
        """
        self.eval()
        device = dna_inputs["input_ids"].device
        
        # Encode DNA
        dna_context = self.encode_dna(dna_inputs)
        
        # Start with a single token (BOS)
        if hasattr(self.text_model.config, 'bos_token_id'):
            bos_token_id = self.text_model.config.bos_token_id
        else:
            bos_token_id = 0  # Default to 0 if not specified
            
        input_ids = torch.full(
            (dna_inputs["input_ids"].size(0), 1),
            bos_token_id,
            device=device
        )
        
        # Autoregressive generation
        for _ in range(max_length - 1):
            # Create attention mask for current sequence
            attention_mask = torch.ones_like(input_ids)
            
            # Encode text
            text_hidden = self.encode_text({"input_ids": input_ids, "attention_mask": attention_mask})
            
            # Apply cross-attention and transformer blocks
            hidden_states = text_hidden
            for block, cross_block in zip(self.gpt_blocks, self.cross_blocks):
                if hasattr(block, '__call__'):
                    hidden_states = block(hidden_states)[0]
                else:
                    hidden_states = block(hidden_states, attention_mask=None)[0]
                    
                if cross_block is not None:
                    hidden_states = cross_block(hidden_states, dna_context, dna_inputs["attention_mask"])
            
            # Get next token logits
            logits = self.lm_head(self.ln_f(hidden_states[:, -1])) / temperature
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if EOS token is generated
            if (next_token == self.text_model.config.eos_token_id).all():
                break
                
        return input_ids
    
    @torch.no_grad()
    def generate_dna_from_text(self, text_inputs, max_length=512, temperature=0.8):
        """
        Generate DNA sequences from text inputs.
        
        Args:
            text_inputs: Dictionary with text input_ids and attention_mask
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            
        Returns:
            Generated DNA token IDs
        """
        self.eval()
        text_hidden = self.encode_text(text_inputs)
        
        # Project text to DNA space
        text_in_dna_space = self.text_to_dna_proj(text_hidden)
        
        # Start with a single token (BOS)
        bos_token_id = self.dna_model.config.bos_token_id if hasattr(self.dna_model.config, 'bos_token_id') else 0
        input_ids = torch.full(
            (text_inputs["input_ids"].size(0), 1),
            bos_token_id,
            dtype=torch.long,
            device=text_inputs["input_ids"].device
        )
        
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
            
            # Stop if EOS token is generated
            eos_token_id = self.dna_model.config.eos_token_id if hasattr(self.dna_model.config, 'eos_token_id') else None
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            
        return input_ids