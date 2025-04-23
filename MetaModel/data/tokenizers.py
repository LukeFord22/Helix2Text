"""Tokenizer utilities for DNA and text sequences."""

from transformers import AutoTokenizer

def load_tokenizers(dna_model_path, text_model_path):
    """
    Load tokenizers for DNA and text models.
    
    Args:
        dna_model_path: Path to DNA model 
        text_model_path: Path to text model
        
    Returns:
        Tuple of (dna_tokenizer, text_tokenizer)
    """
    dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_path, trust_remote_code=True)
    
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_path, trust_remote_code=True)
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token
        text_tokenizer.pad_token_id = text_tokenizer.eos_token_id
    text_tokenizer.padding_side = 'left'
    
    return dna_tokenizer, text_tokenizer