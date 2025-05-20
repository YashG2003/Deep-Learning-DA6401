import torch
from torch.utils.data import Dataset

class TransliterationDataset(Dataset):
    def __init__(self, file_path, source_vocab=None, target_vocab=None, build_vocab=False):
        self.data = []
        
        # Special tokens
        self.SOW = '>'  # Start of word
        self.EOW = '<'  # End of word
        self.PAD = '.'  # Padding
        self.UNK = '?'  # Unknown
        
        # Special token indices
        self.SOW_idx = 0
        self.EOW_idx = 1
        self.PAD_idx = 2
        self.UNK_idx = 3
        
        # Read TSV file
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    target, source = parts[0], parts[1]
                    self.data.append((source, target))
        
        # Build or use provided vocabularies
        if build_vocab:
            source_chars = set()
            target_chars = set()
            
            for source, target in self.data:
                source_chars.update(source)
                target_chars.update(target)
            
            # Create vocabularies
            self.source_char2int = {char: i + 4 for i, char in enumerate(sorted(source_chars))}
            self.target_char2int = {char: i + 4 for i, char in enumerate(sorted(target_chars))}
            
            # Add special tokens
            for token, idx in [(self.SOW, self.SOW_idx), (self.EOW, self.EOW_idx), 
                              (self.PAD, self.PAD_idx), (self.UNK, self.UNK_idx)]:
                self.source_char2int[token] = idx
                self.target_char2int[token] = idx
            
            # Create reverse vocabularies
            self.source_int2char = {i: char for char, i in self.source_char2int.items()}
            self.target_int2char = {i: char for char, i in self.target_char2int.items()}
        else:
            self.source_char2int = source_vocab[0]
            self.target_char2int = target_vocab[0]
            self.source_int2char = source_vocab[1]
            self.target_int2char = target_vocab[1]
        
        # Calculate max length
        self.max_length = max([len(source) for source, _ in self.data] + 
                             [len(target) for _, target in self.data]) + 2  # +2 for SOW and EOW
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        source, target = self.data[idx]
        
        # Convert source to indices
        source_indices = [self.source_char2int.get(char, self.UNK_idx) for char in source]
        source_indices.append(self.EOW_idx)  # Add EOW
        source_indices.extend([self.PAD_idx] * (self.max_length - len(source_indices)))
        
        # Convert target to indices
        target_indices = [self.target_char2int.get(char, self.UNK_idx) for char in target]
        
        # Create decoder input (add SOW at beginning)
        decoder_input = [self.SOW_idx] + target_indices
        decoder_input.extend([self.PAD_idx] * (self.max_length - len(decoder_input)))
        
        # Create decoder target (add EOW at end)
        decoder_target = target_indices + [self.EOW_idx]
        decoder_target.extend([self.PAD_idx] * (self.max_length - len(decoder_target)))
        
        return {
            'source': torch.tensor(source_indices, dtype=torch.long),
            'decoder_input': torch.tensor(decoder_input, dtype=torch.long),
            'decoder_target': torch.tensor(decoder_target, dtype=torch.long),
            'source_text': source,
            'target_text': target
        }
    
    def get_vocab(self):
        return (self.source_char2int, self.source_int2char), (self.target_char2int, self.target_int2char)
