import torch
from ..data.dataset import get_data_loaders
from ..models.dna_text_fusion import DNATextFusion, load_models
from .optimization import create_optimizer_and_scheduler

def setup_training(config):
    """
    Unified setup function that abstracts away DDP/DeepSpeed/SingleGPU logic.
    
    Returns:
        model, optimizer, scheduler, train_loader, test_loader, device, is_main_process, local_rank, dna_tokenizer, text_tokenizer
    """
    using_deepspeed = hasattr(config, 'deepspeed') and config.deepspeed.use_deepspeed
    using_ddp = hasattr(config, 'ddp_runtime') and config.ddp_runtime is not None
    using_distributed = hasattr(config, 'distributed') and config.distributed.get('enabled', False)

    if using_deepspeed:
        model_engine = config.deepspeed_runtime.get('model_engine')
        model = model_engine.module if hasattr(model_engine, 'module') else model_engine
        optimizer = model_engine.optimizer
        scheduler = None
        dna_tokenizer = config.deepspeed_runtime.get('dna_tokenizer')
        text_tokenizer = config.deepspeed_runtime.get('text_tokenizer')
        train_loader = config.deepspeed_runtime.get('train_loader')
        test_loader = config.deepspeed_runtime.get('test_loader')
        device = model_engine.device
        local_rank = config.deepspeed_runtime.get('local_rank', -1)
        is_main_process = local_rank <= 0

    elif using_ddp:
        model = config.ddp_runtime.get('model')
        dna_tokenizer = config.ddp_runtime.get('dna_tokenizer')
        text_tokenizer = config.ddp_runtime.get('text_tokenizer')

        train_loader, test_loader = get_data_loaders(
            dna_tokenizer=dna_tokenizer,
            text_tokenizer=text_tokenizer,
            config=config,
            rank=config.ddp_runtime.get('rank', -1),
            world_size=config.ddp_runtime.get('world_size', 1),
            use_distributed_sampler=True
        )
        optimizer, scheduler = create_optimizer_and_scheduler(
            model=model,
            lr=config.training.learning_rate,
            steps_per_epoch=len(train_loader),
            num_epochs=config.training.num_epochs,
            grad_accum_steps=config.training.grad_accum_steps
        )
        device = config.ddp_runtime.get('device', torch.device('cuda'))
        local_rank = config.ddp_runtime.get('rank', -1)
        is_main_process = (local_rank == 0)

    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = -1
        is_main_process = True

        dna_tokenizer, text_tokenizer, dna_model, text_model = load_models(
            dna_model_path=config.model.dna_model_path,
            text_model_path=config.model.text_model_path,
            device=device
        )

        train_loader, test_loader = get_data_loaders(
            dna_tokenizer=dna_tokenizer,
            text_tokenizer=text_tokenizer,
            config=config
        )

        model = DNATextFusion(
            dna_model=dna_model,
            text_model=text_model,
            cross_attn_interval=config.model.cross_attn_interval
        ).to(device)

        optimizer, scheduler = create_optimizer_and_scheduler(
            model=model,
            lr=config.training.learning_rate,
            steps_per_epoch=len(train_loader),
            num_epochs=config.training.num_epochs,
            grad_accum_steps=config.training.grad_accum_steps
        )

    return model, optimizer, scheduler, train_loader, test_loader, device, is_main_process, local_rank, dna_tokenizer, text_tokenizer