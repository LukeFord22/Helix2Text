"""DNA-Text fusion model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import CrossModalityTransformerBlock
from .components import LearnableFusionBlock, ModalityAdapter

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

        # Enable memory-efficient attention (xformers if available)
        if hasattr(dna_model, 'config'):
            dna_model.config.use_cache = False
            # Try to enable flash attention if available
            try:
                if hasattr(dna_model, 'enable_flash_attention'):
                    dna_model.enable_flash_attention()
            except:
                pass
                
        if hasattr(text_model, 'config'):
            text_model.config.use_cache = False
            # Try to enable flash attention if available
            try:
                if hasattr(text_model, 'enable_flash_attention'):
                    text_model.enable_flash_attention()
            except:
                pass
            
        # Enable gradient checkpointing with more aggressive offloading
        dna_model.gradient_checkpointing_enable()
        text_model.gradient_checkpointing_enable()
        
        # Modality-specific adapters
        self.dna_adapter = ModalityAdapter(dna_hidden)
        self.text_adapter = ModalityAdapter(text_hidden)
        
        # Cross-modality projections with more lightweight options
        self.dna_to_text_proj = nn.Linear(dna_hidden, text_hidden, bias=False)  
        self.text_to_dna_proj = nn.Linear(text_hidden, dna_hidden, bias=False)  
        
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
            CrossModalityTransformerBlock(
                dim=text_hidden,
                cross_dim=dna_hidden,
                n_heads=text_model.config.n_head if hasattr(text_model.config, 'n_head') else 8
            )
            for _ in range(len(self.gpt_blocks))
        ])
        
        # Enable memory-efficient attention for all cross-attention blocks
        for block in self.cross_blocks:
            if block is not None:
                # Try to enable flash attention for cross-attention if available
                try:
                    if hasattr(block.cross_attn, '_use_flash_attention'):
                        block.cross_attn._use_flash_attention = True
                except:
                    pass
        
        # Fusion block for deeper integration
        self.fusion_block = LearnableFusionBlock(
            hidden_dim=text_hidden,
            input_dim_x=text_hidden,
            input_dim_y=dna_hidden
        )
        
        # Project DNA before joint embedding
        self.project_dna_for_joint = nn.Linear(dna_hidden, text_hidden, bias=False)  # Remove bias for efficiency

        # Unified contrastive projection head with more efficient design
        self.joint_projection = nn.Sequential(
            nn.Linear(text_hidden * 2, text_hidden),
            nn.GELU(),
            nn.LayerNorm(text_hidden)
        )
        
        # Two separate LM heads based on model types
        if hasattr(text_model, 'transformer'):
            self.ln_f = text_model.transformer.ln_f
            self.text_lm_head = text_model.lm_head  # For generating text
        elif hasattr(text_model, 'biogpt'):
            self.ln_f = text_model.biogpt.layer_norm
            self.text_lm_head = text_model.output_projection if hasattr(text_model, 'output_projection') else text_model.biogpt.output_projection
            
        # DNA LM head - use the same structure as the text model's head but with DNA vocab size
        if hasattr(dna_model, 'lm_head'):
            self.dna_lm_head = dna_model.lm_head  # For generating DNA
        else:
            # Create a new head for DNA if not available
            # First get DNA vocab size
            dna_vocab_size = dna_model.config.vocab_size
            self.dna_lm_head = nn.Linear(text_hidden, dna_vocab_size)  # From text to DNA
    
    def encode_dna(self, dna_inputs):
        """Process DNA inputs through the DNA model."""
        input_ids = dna_inputs["input_ids"]
        attention_mask = dna_inputs["attention_mask"]
        
        # Build flash-style attention mask
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()

        position_ids = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0)

        dna_outputs = self.dna_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True
        )

        dna_hidden = dna_outputs.hidden_states[-1]
        adapted_dna = self.dna_adapter(dna_hidden.to(dtype=next(self.dna_adapter.parameters()).dtype))
        return adapted_dna
    
    def encode_text(self, text_inputs):
        """Encode text inputs."""
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]
        
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()
        position_ids = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0)

        if hasattr(self.text_model, 'transformer'):
            text_outputs = self.text_model.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True
            )
        else:
            print("fixme")
            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True
            )

        text_hidden = text_outputs.hidden_states[-1]
        adapted_text = self.text_adapter(text_hidden.to(dtype=next(self.text_adapter.parameters()).dtype))
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
        # Encode DNA and Text
        dna_context = self.encode_dna(dna_inputs)
        text_hidden = self.encode_text(text_inputs)

        # Backup input_ids if they require gradients (e.g., for certain loss functions)
        input_ids_backup = {}
        for modality, inputs in (("dna", dna_inputs), ("text", text_inputs)):
            if "input_ids" in inputs:
                input_ids_backup[modality] = inputs["input_ids"].detach().clone() if inputs["input_ids"].requires_grad else None
                del inputs["input_ids"]

        hidden_states = text_hidden

        # Apply CrossModalityTransformer blocks
        for block in self.cross_blocks:
            hidden_states = checkpoint(block, hidden_states, dna_context)

        # Pooled representations
        def masked_mean_pooling(hidden, mask):
            if mask is not None:
                expanded_mask = mask.unsqueeze(-1).expand_as(hidden).float()
                summed = (hidden * expanded_mask).sum(dim=1)
                mean_pooled = summed / expanded_mask.sum(dim=1).clamp(min=1e-9)
                return mean_pooled
            return hidden.mean(dim=1)

        text_pooled = masked_mean_pooling(hidden_states, text_inputs.get("attention_mask"))
        dna_pooled = masked_mean_pooling(dna_context, dna_inputs.get("attention_mask"))

        # Fusion and contrastive joint embedding
        dna_pooled_proj = self.project_dna_for_joint(dna_pooled)
        fused_representation = self.fusion_block(text_pooled, dna_pooled)
        joint_embedding = self.joint_projection(torch.cat([text_pooled, dna_pooled_proj], dim=-1))

        # Logits generation
        text_informed_by_dna = self.ln_f(hidden_states)
        dna_informed_by_text = self.text_to_dna_proj(hidden_states)

        text_logits = self.text_lm_head(text_informed_by_dna)
        dna_logits = self.dna_lm_head(dna_informed_by_text)

        # Clean up
        del hidden_states, text_hidden
        del input_ids_backup
        if not dna_context.requires_grad:
            del dna_context

        # Restore input_ids if needed (optional safety step)
        for modality, backup in input_ids_backup.items():
            if backup is not None:
                if modality == "dna":
                    dna_inputs["input_ids"] = backup
                elif modality == "text":
                    text_inputs["input_ids"] = backup

        return {
            "text_logits": text_logits,
            "dna_logits": dna_logits,
            "joint_embedding": joint_embedding,
            "fused_representation": fused_representation,
            "text_pooled": text_pooled,
            "dna_pooled": dna_pooled,
        }
    
    @torch.no_grad()
    def generate_text_from_dna(self, dna_inputs, max_length=128, temperature=0.8, top_k=50, top_p=0.9):
        """
        Generate text from DNA inputs with optimized memory usage.
        
        Args:
            dna_inputs: Dictionary with DNA input_ids and attention_mask
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Probability threshold for nucleus sampling (cumulative probability)
            
        Returns:
            Generated text token IDs
        """
        self.eval()
        device = dna_inputs["input_ids"].device
        
        # Encode DNA once to avoid repeated computation
        dna_context = self.encode_dna(dna_inputs)
        
        # Determine start token
        if hasattr(self.text_model.config, 'bos_token_id'):
            bos_token_id = self.text_model.config.bos_token_id
        else:
            bos_token_id = 0  # Default to 0 if not specified
            
        # Get batch size from input
        batch_size = dna_inputs["input_ids"].size(0)
        
        # Start with beginning of sequence token
        input_ids = torch.full(
            (batch_size, 1),
            bos_token_id,
            device=device,
            dtype=torch.long
        )
        
        # Track attention for more efficient generation by reusing context
        past_key_values = None
        curr_length = 1
        
        # Keep track of which sequences are finished
        unfinished_sequences = torch.ones(batch_size, device=device, dtype=torch.long)
        
        # Create original attention mask (all valid at the start)
        attention_mask = torch.ones_like(input_ids)
        
        # Get EOS token for early stopping
        eos_token_id = self.text_model.config.eos_token_id if hasattr(self.text_model.config, 'eos_token_id') else None
        
        # Set up memory tracking if in debug mode
        debug_memory = False
        if debug_memory:
            memory_before = torch.cuda.memory_allocated() / (1024**2)
            print(f"Memory before generation: {memory_before:.2f} MB")
        
        # Autoregressive generation
        while curr_length < max_length:
            # Encode text - only encode the newly added token by using past_key_values
            if hasattr(self.text_model, 'transformer'):
                # GPT-2 style

                position_ids = torch.arange(0, input_ids.size(1), device=device).unsqueeze(0)
                text_outputs = self.text_model.transformer(
                    input_ids=input_ids[:, -1:] if past_key_values is not None else input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids[:, -1:] if past_key_values is not None else position_ids,
                    output_hidden_states=True,
                    use_cache=True,
                    past_key_values=past_key_values
                )
                hidden_states = text_outputs.hidden_states[-1]
                past_key_values = text_outputs.past_key_values
            else:
                # BioGPT style - may need to adapt based on specific model
                print("fixme")
                text_outputs = self.text_model(
                    input_ids=input_ids[:, -1:] if past_key_values is not None else input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=True,
                    past_key_values=past_key_values
                )
                hidden_states = text_outputs.hidden_states[-1]
                past_key_values = text_outputs.past_key_values
            
            # Apply cross-attention from DNA to the last token's representation
            for block, cross_block in zip(self.gpt_blocks, self.cross_blocks):
                if cross_block is not None:
                    hidden_states = cross_block(
                        hidden_states, dna_context, dna_inputs["attention_mask"]
                    )
            
            # Get next token logits - only need the last position
            next_token_hidden = self.ln_f(hidden_states[:, -1, :])
            next_token_logits = self.text_lm_head(next_token_hidden) / temperature
            
            # Apply filtering for top-k sampling
            if top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))  # Safety check
                # Remove all tokens with a probability less than the last token of the top-k
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))

            # Apply nucleus (top-p) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))
            
            # Convert logits to probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Mark sequences as finished if they generated EOS
            if eos_token_id is not None:
                # Update the unfinished_sequences tensor for sequences that hit EOS
                unfinished_sequences = unfinished_sequences.mul((next_token != eos_token_id).long())
                
                # Check if all sequences are finished
                if unfinished_sequences.max() == 0:
                    break
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([
                attention_mask, 
                attention_mask.new_ones((batch_size, 1))
            ], dim=1)
            curr_length += 1
        
        # Track memory usage if in debug mode
        if debug_memory:
            memory_after = torch.cuda.memory_allocated() / (1024**2)
            print(f"Memory after generation: {memory_after:.2f} MB")
            print(f"Memory used during generation: {memory_after - memory_before:.2f} MB")
        
        # Clean up
        del dna_context, hidden_states, past_key_values
        torch.cuda.empty_cache()
                
        return input_ids
    
    @torch.no_grad()
    def generate_dna_from_text(self, text_inputs, max_length=512, temperature=0.8, top_k=50, top_p=0.9):
        """
        Generate DNA sequences from text inputs with memory-efficient decoding.
        
        Args:
            text_inputs: Dictionary with text input_ids and attention_mask
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Probability threshold for nucleus sampling (cumulative probability)
            
        Returns:
            Generated DNA token IDs
        """
        self.eval()
        device = text_inputs["input_ids"].device
        batch_size = text_inputs["input_ids"].size(0)
        
        # Encode text - do this once to avoid repeated computation
        text_hidden = self.encode_text(text_inputs)
        
        # Project text to DNA space
        text_in_dna_space = self.text_to_dna_proj(text_hidden)
        
        # Determine start token
        if hasattr(self.dna_model.config, 'bos_token_id'):
            bos_token_id = self.dna_model.config.bos_token_id
        else:
            bos_token_id = 0  # Default to 0 if not specified
            
        # Start with beginning of sequence token
        input_ids = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Create original attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # Keep track of which sequences are finished
        unfinished_sequences = torch.ones(batch_size, device=device, dtype=torch.long)
        
        # Get EOS token for early stopping
        eos_token_id = self.dna_model.config.eos_token_id if hasattr(self.dna_model.config, 'eos_token_id') else None
        
        # Track past key values for more efficient generation
        past_key_values = None
        curr_length = 1
        
        # Generate DNA sequence token by token
        while curr_length < max_length:
            # Get DNA model outputs for current sequence using efficient caching
            if past_key_values is not None:
                # If we have past key values, only process the last token
                current_token = input_ids[:, -1:]
                current_mask = attention_mask
            else:
                # First iteration - process the full sequence (just the BOS token)
                current_token = input_ids
                current_mask = attention_mask
            
            # Forward pass
            dna_outputs = self.dna_model(
                input_ids=current_token,
                attention_mask=current_mask,
                encoder_hidden_states=text_in_dna_space,  # Condition on text
                encoder_attention_mask=text_inputs["attention_mask"],
                use_cache=True,
                past_key_values=past_key_values,
                output_hidden_states=True
            )
            
            # Update past key values for next iteration
            past_key_values = dna_outputs.past_key_values
            
            # Get logits for next token prediction
            next_token_logits = dna_outputs.logits[:, -1, :] / temperature
            
            # Apply filtering for top-k sampling
            if top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))  # Safety check
                # Remove all tokens with a probability less than the last token of the top-k
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))

            # Apply nucleus (top-p) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))
            
            # Convert logits to probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Mark sequences as finished if they generated EOS
            if eos_token_id is not None:
                # Update the unfinished_sequences tensor for sequences that hit EOS
                unfinished_sequences = unfinished_sequences.mul((next_token != eos_token_id).long())
                
                # Check if all sequences are finished
                if unfinished_sequences.max() == 0:
                    break
            
            # Append to the sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([
                attention_mask, 
                attention_mask.new_ones((batch_size, 1))
            ], dim=1)
            curr_length += 1
        
        # Clean up
        del text_hidden, text_in_dna_space, past_key_values
        torch.cuda.empty_cache()
            
        return input_ids