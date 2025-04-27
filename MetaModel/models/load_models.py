import gc
import os
import json
import torch
import deepspeed
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


from ..data.dataset import get_data_loaders
from ..models.dna_text_fusion import DNATextFusion
from ..training.optimization import create_optimizer_and_scheduler

def load_ds_config():
    ds_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "ds.config.json")
    with open(ds_config_path, "r") as f:
        return json.load(f)

DS_CONFIG = load_ds_config()

def load_models(dna_model_path, text_model_path):
    """
    Load pretrained DNA and text models with memory optimization.
    
    Args:
        dna_model_path: Path to the DNA model
        text_model_path: Path to the text model
        device: Device to load models to
        use_half: Whether to use half precision (fp16)
        
    Returns:
        Tuple of (dna_tokenizer, text_tokenizer, dna_model, text_model)
    """
    # Load tokenizers
    dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_path, trust_remote_code=True)
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_path, trust_remote_code=True)
    
    
    # Load DNA model with memory optimizations
    dna_model = AutoModelForCausalLM.from_pretrained(
        dna_model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None,  # <--- Stay on CPU!
        attn_implementation="flash_attention_2",  # If supported
    )
    
    # Load text model with memory optimizations but WITHOUT flash attention for BioGPT
    text_model = AutoModelForCausalLM.from_pretrained(
        text_model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None,
        attn_implementation="flash_attention_2",
    )
    
    # Enable gradient checkpointing immediately after loading
    dna_model.gradient_checkpointing_enable()
    text_model.gradient_checkpointing_enable()
    
    # Garbage collect to free memory
    gc.collect()
    torch.cuda.empty_cache()
    
    return dna_tokenizer, text_tokenizer, dna_model, text_model


def setup_fusion_and_deepspeed(rank, world_size, config):
    """Loads models, creates fusion, initializes DeepSpeed."""

    # Load models on CPU
    dna_tokenizer, text_tokenizer, dna_model, text_model = load_models(
        config.model.dna_model_path,
        config.model.text_model_path,
    )

    # Fusion model still on CPU here
    fusion_model = DNATextFusion(
        dna_model=dna_model,
        text_model=text_model,
        cross_attn_interval=config.model.cross_attn_interval
    )

    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Rank {rank}] Model fusion complete. Allocated CPU RAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


    # create the loaders
    train_loader, test_loader = get_data_loaders(
    dna_tokenizer, text_tokenizer, 
    rank=rank, world_size=world_size
    )

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
    model=fusion_model,
    lr=config.training.learning_rate,
    steps_per_epoch=len(train_loader),  
    num_epochs=config.training.num_epochs,
    grad_accum_steps=config.training.grad_accum_steps,
    )

    # Assume config is your TrainingConfig instance
    DS_CONFIG["train_batch_size"] = config.batch_size
    DS_CONFIG["train_micro_batch_size_per_gpu"] = config.batch_size // config.grad_accum_steps
    DS_CONFIG["gradient_accumulation_steps"] = config.grad_accum_steps

    # DeepSpeed initialization
    model_engine, _, _, _ = deepspeed.initialize(
        model=fusion_model,
        model_parameters=fusion_model.parameters(),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=DS_CONFIG,
        custom_dataloader=train_loader,      
        custom_eval_dataloader=test_loader    
    )

    print(f"[Rank {rank}] After DeepSpeed init: GPU memory {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return model_engine, dna_tokenizer, text_tokenizer