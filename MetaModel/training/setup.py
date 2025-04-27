def setup_training(config):
    """
    Prepares the model, optimizer, scheduler, dataloaders, device, and ranks.
    """

    # Load model and device configuration
    model_engine = config.deepspeed_runtime['model_engine']
    device = config.deepspeed_runtime['device']
    local_rank = config.deepspeed_runtime.get('rank', -1)

    # Fetch the tokenizer from the config
    dna_tokenizer = config.deepspeed_runtime['dna_tokenizer']
    text_tokenizer = config.deepspeed_runtime['text_tokenizer']

    # Fetch optimizer and scheduler from model_engine 
    optimizer = model_engine.optimizer
    scheduler = model_engine.lr_scheduler

    # Fetch data from model_engine
    train_loader = model_engine.train_dataloader
    test_loader = model_engine.eval_dataloader

    return model_engine, optimizer, scheduler, train_loader, test_loader, device, local_rank, dna_tokenizer, text_tokenizer

