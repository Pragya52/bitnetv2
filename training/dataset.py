"""
Dataset handling for BitNet v2 training
Supports RedPajama and other large-scale text datasets
"""

import torch
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset
import random
from typing import Dict, List, Optional

class RedPajamaDataset(IterableDataset):
    """Dataset wrapper for RedPajama training data"""
    
    def __init__(self, tokenizer, max_length=2048, split='train', streaming=True, subset=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.streaming = streaming
        
        # Load RedPajama dataset (or substitute with another dataset)
        try:
            if subset:
                # Load a specific subset for faster testing
                self.dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", 
                                          split=f"{split}[:{subset}]", streaming=streaming)
            else:
                self.dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", 
                                          split=split, streaming=streaming)
        except:
            # Fallback to a smaller dataset for testing
            print("Warning: Using C4 dataset instead of RedPajama")
            if subset:
                self.dataset = load_dataset("c4", "en", split=f"{split}[:{subset}]", streaming=streaming)
            else:
                self.dataset = load_dataset("c4", "en", split=split, streaming=streaming)
        
        if streaming:
            self.iterator = iter(self.dataset)
        else:
            self.data = list(self.dataset)
    
    def __iter__(self):
        if self.streaming:
            return self
        else:
            return iter(self._process_data())
    
    def __next__(self):
        if self.streaming:
            try:
                item = next(self.iterator)
                return self._process_item(item)
            except StopIteration:
                # Reset iterator when dataset is exhausted
                self.iterator = iter(self.dataset)
                item = next(self.iterator)
                return self._process_item(item)
        else:
            raise NotImplementedError("Non-streaming mode not implemented for __next__")
    
    def _process_data(self):
        """Process all data for non-streaming mode"""
        for item in self.data:
            processed = self._process_item(item)
            if processed is not None:
                yield processed
    
    def _process_item(self, item):
        """Process a single data item"""
        text = item['text']
        
        # Tokenize
        tokens = self.tokenizer.encode(
            text, 
            max_length=self.max_length, 
            truncation=True,
            padding=False,
            add_special_tokens=True
        )
        
        if len(tokens) < 2:
            return None  # Skip short sequences
        
        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'labels': torch.tensor(tokens[1:], dtype=torch.long)
        }
    
    def __len__(self):
        if self.streaming:
            return float('inf')  # Streaming datasets have infinite length
        else:
            return len(self.data)

class C4Dataset(Dataset):
    """C4 dataset for evaluation"""
    
    def __init__(self, tokenizer, max_length=2048, split='validation', max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load C4 dataset
        dataset = load_dataset("c4", "en", split=split, streaming=False)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.data = []
        for item in dataset:
            processed = self._process_item(item)
            if processed is not None:
                self.data.append(processed)
    
    def _process_item(self, item):
        """Process a single data item"""
        text = item['text']
        
        # Tokenize
        tokens = self.tokenizer.encode(
            text, 
            max_length=self.max_length, 
            truncation=True,
            padding=False,
            add_special_tokens=True
        )
        
        if len(tokens) < 2:
            return None
        
        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'labels': torch.tensor(tokens[1:], dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class WikiTextDataset(Dataset):
    """WikiText dataset for evaluation"""
    
    def __init__(self, tokenizer, max_length=2048, split='test', version='wikitext-2-raw-v1'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load WikiText dataset
        dataset = load_dataset("wikitext", version, split=split)
        
        self.data = []
        for item in dataset:
            processed = self._process_item(item)
            if processed is not None:
                self.data.append(processed)
    
    def _process_item(self, item):
        """Process a single data item"""
        text = item['text']
        
        # Skip empty lines and headers
        if not text.strip() or text.strip().startswith('='):
            return None
        
        # Tokenize
        tokens = self.tokenizer.encode(
            text, 
            max_length=self.max_length, 
            truncation=True,
            padding=False,
            add_special_tokens=True
        )
        
        if len(tokens) < 2:
            return None
        
        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'labels': torch.tensor(tokens[1:], dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """Collate function for DataLoader"""
    # Filter out None items
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    max_len = max(len(item['input_ids']) for item in batch)
    
    input_ids = []
    labels = []
    
    for item in batch:
        # Pad sequences
        input_len = len(item['input_ids'])
        pad_len = max_len - input_len
        
        input_ids.append(torch.cat([
            item['input_ids'],
            torch.full((pad_len,), 0, dtype=torch.long)  # Pad with 0
        ]))
        
        labels.append(torch.cat([
            item['labels'],
            torch.full((pad_len,), -100, dtype=torch.long)  # Ignore padding in loss
        ]))
    
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels)
    }

def get_dataset(dataset_name: str, tokenizer, split: str = 'train', **kwargs):
    """Factory function to get datasets"""
    
    if dataset_name.lower() == 'redpajama':
        return RedPajamaDataset(tokenizer, split=split, **kwargs)
    elif dataset_name.lower() == 'c4':
        return C4Dataset(tokenizer, split=split, **kwargs)
    elif dataset_name.lower() == 'wikitext':
        return WikiTextDataset(tokenizer, split=split, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

class MultiDatasetWrapper(IterableDataset):
    """Wrapper to combine multiple datasets with sampling weights"""
    
    def __init__(self, datasets: Dict[str, Dataset], weights: Optional[Dict[str, float]] = None):
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())
        
        if weights is None:
            weights = {name: 1.0 for name in self.dataset_names}
        
        # Normalize weights
        total_weight = sum(weights.values())
        self.weights = {name: weights[name] / total_weight for name in self.dataset_names}
        
        # Create iterators for streaming datasets
        self.iterators = {}
        for name, dataset in datasets.items():
            if hasattr(dataset, '__iter__'):
                self.iterators[name] = iter(dataset)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Sample a dataset based on weights
        dataset_name = random.choices(self.dataset_names, 
                                    weights=[self.weights[name] for name in self.dataset_names])[0]
        
        dataset = self.datasets[dataset_name]
        
        if dataset_name in self.iterators:
            # Streaming dataset
            try:
                return next(self.iterators[dataset_name])
            except StopIteration:
                # Reset iterator
                self.iterators[dataset_name] = iter(dataset)
                return next(self.iterators[dataset_name])
        else:
            # Regular dataset
            idx = random.randint(0, len(dataset) - 1)
            return dataset[idx]
