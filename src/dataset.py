import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TextGenerationDataset(Dataset):
    def __init__(self, X, Y, vocab, max_seq_length=50):
        self.X = X
        self.y = Y
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.vocab_size = len(vocab)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        input_tokens = self.X[idx]
        target_token = self.y[idx]

        input_ids = self.tokens_to_indices(input_tokens, self.max_seq_length)
        target_id = self.vocab.get(target_token, self.vocab['<UNK>'])

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target': torch.tensor(target_id, dtype=torch.long)
        }
    
    def tokens_to_indices(self, tokens, max_length):
        indices = []
        for token in tokens:
            indices.append(self.vocab.get(token, self.vocab['<UNK>']))
        
        if len(indices) > max_length:
            indices = indices[:max_length]
        else:
            indices = indices + [self.vocab['<PAD>']] * (max_length - len(indices))
        
        return indices

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, vocab, batch_size=32, max_seq_length=50):
    train_dataset = TextGenerationDataset(X_train, y_train, vocab, max_seq_length)
    val_dataset = TextGenerationDataset(X_val, y_val, vocab, max_seq_length)
    test_dataset = TextGenerationDataset(X_test, y_test, vocab, max_seq_length)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader