"""Dataset creation and processing for DNA-Text fusion models."""

import os
import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import IterableDataset
import torch
import numpy as np

from ..configs.default_config import default_config

class StreamingTokenizedDataset(IterableDataset):
    def __init__(self, dataframe, dna_tokenizer, text_tokenizer, config, split="train"):
        self.data = dataframe
        self.dna_tokenizer = dna_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_length = config.training.max_seq_length
        self.split = split
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.data)
        else:
            per_worker = int(np.ceil(len(self.data) / float(worker_info.num_workers)))
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, len(self.data))
        
        for idx in range(iter_start, iter_end):
            row = self.data.iloc[idx]
            dna = row["DNA_Sequence"]
            text = row["text"]

            dna_enc = self.dna_tokenizer(
                dna,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            text_enc = self.text_tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )

            yield {
                "dna_input_ids": dna_enc.input_ids.squeeze(0),
                "dna_attention_mask": dna_enc.attention_mask.squeeze(0),
                "text_input_ids": text_enc.input_ids.squeeze(0),
                "text_attention_mask": text_enc.attention_mask.squeeze(0)
            }
    def __len__(self):
        return len(self.data)

def get_data_loaders(dna_tokenizer, text_tokenizer, config, rank=None, world_size=None):
    df = pd.read_csv(config.data.local_path)

    train_data_fraction = getattr(config.data, 'train_data_fraction', 0.9)
    train_split_fraction = getattr(config.data, 'train_split_fraction', 1)

    sampled_df = df.sample(frac=train_data_fraction, random_state=42)
    train_df = sampled_df.sample(frac=train_split_fraction, random_state=42)
    test_df = sampled_df.drop(train_df.index)

    train_dataset = StreamingTokenizedDataset(train_df, dna_tokenizer, text_tokenizer, config, split="train")
    test_dataset = StreamingTokenizedDataset(test_df, dna_tokenizer, text_tokenizer, config, split="test")

    # No samplers anymore 
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,         # You CANNOT shuffle an IterableDataset
        pin_memory=True,
        num_workers=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    return train_loader, test_loader