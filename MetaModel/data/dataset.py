"""Dataset creation and processing for DNA-Text fusion models."""

import os
import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict

from ..configs.default_config import default_config

def get_tokenized_dataset(dna_tokenizer, text_tokenizer, config=None):
    """
    Create a tokenized dataset from the DNA-text pairs.
    
    Args:
        dna_tokenizer: Tokenizer for DNA sequences
        text_tokenizer: Tokenizer for text descriptions
        config: Configuration object
        
    Returns:
        DatasetDict containing tokenized training and test datasets
    """
    config = config or default_config
    
    # Check if separate train and eval paths are provided and splitting is not enabled
    if config.data.eval_path and os.path.exists(config.data.eval_path) and not getattr(config.data, 'split_data', True):
        # Use separate files for train and eval
        print(f"Using separate training and evaluation datasets")
        train_df = pd.read_csv(config.data.local_path)
        test_df = pd.read_csv(config.data.eval_path)
    else:
        # Use a single file and split it
        # First check if data.train_data_fraction exists (GUI sets it here), otherwise fallback to training.train_data_fraction
        train_data_fraction = getattr(config.data, 'train_data_fraction', 
                                      getattr(config.training, 'train_data_fraction', 0.9))
        
        # First check if data.train_split_fraction exists (GUI sets it here), otherwise fallback to training.train_split_fraction
        train_split_fraction = getattr(config.data, 'train_split_fraction', 
                                       getattr(config.training, 'train_split_fraction', 1))
        
        print(f"Splitting data from single dataset file:")
        print(f"  - Train data fraction: {train_data_fraction}")
        print(f"  - Train split fraction: {train_split_fraction}")
        
        # Load the data
        df = pd.read_csv(config.data.local_path)
        
        # Sample according to configuration
        sampled_df = df.sample(frac=train_data_fraction, random_state=42)
        train_df = sampled_df.sample(frac=train_split_fraction, random_state=42)
        test_df = sampled_df.drop(train_df.index)
    
    print(f"Training data size: {len(train_df)}, Evaluation data size: {len(test_df)}")
    
    # Create dataset dictionary
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df)
    })
    
    # Define tokenization function
    def tokenize_fn(example):
        dna_enc = dna_tokenizer(
            example["DNA_Sequence"], 
            truncation=True, 
            padding="max_length", 
            max_length=config.training.max_seq_length
        )
        text_enc = text_tokenizer(
            example["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=config.training.max_seq_length
        )
        return {
            "dna_input_ids": dna_enc["input_ids"],
            "dna_attention_mask": dna_enc["attention_mask"],
            "text_input_ids": text_enc["input_ids"],
            "text_attention_mask": text_enc["attention_mask"],
        }
    
    # Apply tokenization
    tokenized_data = dataset.map(tokenize_fn, batched=True)
    tokenized_data.set_format("torch")
    
    return tokenized_data

def get_data_loaders(dna_tokenizer, text_tokenizer, config=None, rank=None, world_size=None, use_distributed_sampler=False):
    """
    Create DataLoaders for training and testing.
    
    Args:
        dna_tokenizer: Tokenizer for DNA sequences
        text_tokenizer: Tokenizer for text descriptions
        config: Configuration object
        rank: Process rank in distributed training
        world_size: Total number of processes in distributed training
        use_distributed_sampler: Whether to use a DistributedSampler
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    config = config or default_config
    tokenized_data = get_tokenized_dataset(dna_tokenizer, text_tokenizer, config)
    
    # Configure samplers and loaders based on distributed setting
    if use_distributed_sampler and rank is not None and world_size is not None:
        from torch.utils.data.distributed import DistributedSampler
        
        # Create distributed samplers
        train_sampler = DistributedSampler(
            tokenized_data['train'],
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42
        )
        
        # For test data, we still want to split it across GPUs
        # but shuffling is not necessary
        test_sampler = DistributedSampler(
            tokenized_data['test'],
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        train_loader = DataLoader(
            tokenized_data['train'], 
            batch_size=config.training.batch_size,
            sampler=train_sampler,  # Use sampler instead of shuffle
            pin_memory=True,
            num_workers=2
        )
        
        test_loader = DataLoader(
            tokenized_data['test'], 
            batch_size=config.training.batch_size,
            sampler=test_sampler,  # Use sampler for test data too
            pin_memory=True,
            num_workers=2
        )
        
        print(f"Created distributed data loaders for rank {rank}/{world_size}")
    else:
        # Standard non-distributed loaders
        train_loader = DataLoader(
            tokenized_data['train'], 
            batch_size=config.training.batch_size, 
            shuffle=True, 
            pin_memory=True
        )
        
        test_loader = DataLoader(
            tokenized_data['test'], 
            batch_size=config.training.batch_size, 
            pin_memory=True
        )
    
    return train_loader, test_loader