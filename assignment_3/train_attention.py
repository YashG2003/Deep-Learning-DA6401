import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import wandb
import random
import numpy as np

from data.dataset import TransliterationDataset
from models.encoder import Encoder
from models.decoder import AttentionDecoder
from models.seq2seq import AttentionSeq2Seq
from utils.train import train, train_model
from utils.evaluate import evaluate, beam_search_evaluate, generate_predictions

def train_attention_model(data_dir):
    # Best hyperparameters from sweep
    config = {
        'batch_size': 64,
        'embedding_size': 128,
        'hidden_size': 512,
        'encoder_layers': 3,
        'decoder_layers': 3,
        'cell_type': 'gru',
        'dropout': 0.3,
        'learning_rate': 5e-4,
        'clip': 1.0,
        'epochs': 10,
        'beam_width': 3,
        'use_wandb': True,
        'model_name': 'best_attention_model'
    }
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_dataset = TransliterationDataset(os.path.join(data_dir, 'hi.translit.sampled.train.tsv'), build_vocab=True)
    val_dataset = TransliterationDataset(os.path.join(data_dir, 'hi.translit.sampled.dev.tsv'), 
                                        train_dataset.get_vocab()[0], train_dataset.get_vocab()[1])
    test_dataset = TransliterationDataset(os.path.join(data_dir, 'hi.translit.sampled.test.tsv'), 
                                         train_dataset.get_vocab()[0], train_dataset.get_vocab()[1])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # Get vocabulary sizes
    source_vocab_size = len(train_dataset.source_char2int)
    target_vocab_size = len(train_dataset.target_char2int)
    
    # Initialize wandb
    if config['use_wandb']:
        wandb.init(project="da6401_a3", name="best_attention_model")
        wandb.config.update(config)
    
    # Create model
    encoder = Encoder(
        input_size=source_vocab_size,
        embedding_size=config['embedding_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['encoder_layers'],
        dropout=config['dropout'] if config['encoder_layers'] > 1 else 0,
        cell_type=config['cell_type']
    )
    
    decoder = AttentionDecoder(
        output_size=target_vocab_size,
        embedding_size=config['embedding_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['decoder_layers'],
        dropout=config['dropout'] if config['decoder_layers'] > 1 else 0,
        cell_type=config['cell_type']
    )
    
    model = AttentionSeq2Seq(encoder, decoder, device).to(device)
    
    # Define optimizer and loss function
    optimizer = optim.NAdam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.NLLLoss(ignore_index=train_dataset.PAD_idx)
    
    # Train model
    best_model_state, train_losses, val_losses, val_accs = train_model(
        model, train_loader, val_loader, optimizer, criterion, config, device
    )
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
    
    # Evaluate with beam search
    beam_acc = beam_search_evaluate(model, test_loader, device, config['beam_width'])
    print(f'Beam Search Test Acc (width={config["beam_width"]}): {beam_acc:.2f}%')
    
    # Log final metrics
    if config['use_wandb']:
        wandb.log({
            'test_loss': test_loss,
            'test_acc': test_acc,
            'beam_search_acc': beam_acc
        })
        wandb.finish()
    
    # Generate predictions on test set
    generate_predictions(model, test_dataset, device, 'predictions_attention', 'attention', config['beam_width'])
    
    return model, train_dataset, test_acc, beam_acc

if __name__ == '__main__':
    train_attention_model('data/dakshina_dataset_v1.0/hi/lexicons/')
