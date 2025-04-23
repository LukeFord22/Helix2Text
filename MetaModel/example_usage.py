"""
This script shows how to use the modular Helix2Text implementation
to train a DNA-Text fusion model with the same functionality as METAGENE-1model.py.
"""

import torch
import os
import argparse
from datetime import datetime

# Import from modular structure
from Helix2Text.MetaModel.configs.default_config import Config
from Helix2Text.MetaModel.data.dataset import download_dataset, get_data_loaders
from Helix2Text.MetaModel.models.attention import NormalizedGatedCrossAttentionBlock
from Helix2Text.MetaModel.models.dna_text_fusion import DNATextFusion, load_models
from Helix2Text.MetaModel.losses.contrastive import DNATextContrastiveLoss
from Helix2Text.MetaModel.losses.criterion import DNATextCriterion
from Helix2Text.MetaModel.training.trainer import DNATextTrainer
from Helix2Text.MetaModel.training.evaluation import evaluate_model
from Helix2Text.MetaModel.utils.optimization import create_optimizer_and_scheduler
from Helix2Text.MetaModel.utils.logging import setup_wandb

def main():
    parser = argparse.ArgumentParser(description="Train DNA-Text fusion model")
    parser.add_argument("--config", type=str, default="Helix2Text/MetaModel/configs/training_config.yaml", 
                        help="Path to config YAML file")
    parser.add_argument("--output_dir", type=str, default="./outputs", 
                        help="Directory to save model outputs")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # === 1. Load Configuration ===
    from Helix2Text.MetaModel.configs.default_config import load_config
    config = load_config(args.config)
    
    # Override with METAGENE-1model.py parameters if needed
    # This preserves compatibility with the original script's parameters
    config.training.batch_size = 2
    config.training.num_epochs = 8
    config.training.learning_rate = 3e-5
    config.training.max_seq_length = 512
    config.training.grad_accum_steps = 5
    config.training.train_data_fraction = 1.0
    config.training.semantic_loss_weight = 0.5
    config.training.frozen_layers = 0
    config.training.early_stop_patience = 3
    
    # === 2. Setup Logging ===
    run = setup_wandb(config)
    experiment_name = f"dna-text-fusion-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"Starting experiment: {experiment_name}")
    
    # === 3. Load Models ===
    print("Loading models and tokenizers...")
    dna_tokenizer, text_tokenizer, dna_model, text_model = load_models(
        dna_model_path=config.model.dna_model_path,
        text_model_path=config.model.text_model_path,
        device=device
    )
    
    # === 4. Load Dataset ===
    print("Preparing datasets...")
    # Using the dataset functions from the modular implementation
    dataset_url = config.data.dataset_url
    local_path = config.data.local_path
    download_dataset(dataset_url, local_path)
    
    train_loader, test_loader = get_data_loaders(
        dna_tokenizer=dna_tokenizer,
        text_tokenizer=text_tokenizer,
        config=config
    )
    
    # === 5. Initialize Model ===
    print("Initializing DNA-Text fusion model...")
    model = DNATextFusion(
        dna_model=dna_model,
        text_model=text_model,
        cross_attn_interval=config.model.cross_attn_interval
    ).to(device)
    
    # Freeze layers if specified
    if config.training.frozen_layers > 0:
        layers_frozen = 0
        # Access transformer blocks in a way compatible with the modular structure
        if hasattr(model.text_model, 'transformer'):
            blocks = model.text_model.transformer.h
        elif hasattr(model.text_model, 'biogpt'):
            blocks = model.text_model.biogpt.layers
        else:
            print("Warning: Couldn't identify transformer blocks to freeze")
            blocks = []
            
        for i, block in enumerate(blocks):
            if i < config.training.frozen_layers:
                for param in block.parameters():
                    param.requires_grad = False
                layers_frozen += 1
        print(f"✅ Frozen {layers_frozen} layers in the text model")
    
    # === 6. Setup Optimizer and Scheduler ===
    print("Setting up optimizer and scheduler...")
    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        lr=config.training.learning_rate,
        steps_per_epoch=len(train_loader),
        num_epochs=config.training.num_epochs,
        grad_accum_steps=config.training.grad_accum_steps
    )
    
    # === 7. Setup Trainer ===
    print("Setting up trainer...")
    trainer = DNATextTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device
    )
    
    # === 8. Train Model ===
    print("Starting training...")
    model, final_metrics = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        text_tokenizer=text_tokenizer,
        dna_tokenizer=dna_tokenizer
    )
    
    # === 9. Final Evaluation ===
    print("\nTraining complete. Running final evaluation...")
    final_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        text_tokenizer=text_tokenizer,
        dna_tokenizer=dna_tokenizer,
        device=device,
        current_epoch="final"
    )
    
    # Print final metrics
    print("\n" + "="*50)
    print("Final Evaluation Results:")
    for metric_name, metric_value in final_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print("="*50)
    
    # === 10. Save Model ===
    model_save_path = os.path.join(args.output_dir, "dna_text_fusion_final.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Provide example commands for using the model
    print("\nTo generate text from DNA sequences:")
    print(f"python -m Helix2Text.MetaModel.scripts.generate --model_path {model_save_path} --mode dna2text --input \"ATGC...\"")
    
    print("\nTo generate DNA from text descriptions:")
    print(f"python -m Helix2Text.MetaModel.scripts.generate --model_path {model_save_path} --mode text2dna --input \"This protein...\"")

if __name__ == "__main__":
    main()