import wandb
import torch
import random
import numpy as np
from torch.utils.data import DataLoader

from data.dataset import TransliterationDataset
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from utils.train import train
from utils.evaluate import evaluate

def run_vanilla_sweep():
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'name': 'vanilla_sweep',
        'metric': {
            'name': 'val_acc',
            'goal': 'maximize'
        },
        'parameters': {
            'batch_size': {'values': [32, 64, 128]},
            'embedding_size': {'values': [64, 128, 256]},
            'hidden_size': {'values': [128, 256, 512]},
            'encoder_layers': {'values': [1, 2, 3]},
            'decoder_layers': {'values': [1, 2, 3]},
            'cell_type': {'values': ['rnn', 'gru', 'lstm']}, 
            'dropout': {'values': [0.2, 0.3]},
            'learning_rate': {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'clip': {'value': 1.0},
            'epochs': {'value': 10},  
            'beam_width': {'values': [1, 3]}
        }
    }
    
    # Initialize wandb
    wandb.login()
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project="da6401_a3")
    
    # Define training function for sweep
    def sweep_train():
        with wandb.init() as run:
            config = wandb.config
            config['use_wandb'] = True
            
            run_name = (
                f"{wandb.config.cell_type}_"
                f"lr_{wandb.config.learning_rate}_"
                f"enc_lay_{wandb.config.encoder_layers}_"
                f"dec_lay_{wandb.config.decoder_layers}"
            )
            wandb.run.name = run_name  # Set the run name
            
            # Set random seeds for reproducibility
            torch.manual_seed(42)
            random.seed(42)
            np.random.seed(42)
            
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load data
            train_dataset = TransliterationDataset('data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv', build_vocab=True)
            val_dataset = TransliterationDataset('data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv', 
                                                train_dataset.get_vocab()[0], train_dataset.get_vocab()[1])
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
            
            # Get vocabulary sizes
            source_vocab_size = len(train_dataset.source_char2int)
            target_vocab_size = len(train_dataset.target_char2int)
            
            # Create model
            encoder = Encoder(
                input_size=source_vocab_size,
                embedding_size=config.embedding_size,
                hidden_size=config.hidden_size,
                num_layers=config.encoder_layers,
                dropout=config.dropout if config.encoder_layers > 1 else 0,
                cell_type=config.cell_type
            )
            
            decoder = Decoder(
                output_size=target_vocab_size,
                embedding_size=config.embedding_size,
                hidden_size=config.hidden_size,
                num_layers=config.decoder_layers,
                dropout=config.dropout if config.decoder_layers > 1 else 0,
                cell_type=config.cell_type
            )
            
            model = Seq2Seq(encoder, decoder, device).to(device)
            
            # Define optimizer and loss function
            optimizer = torch.optim.NAdam(model.parameters(), lr=config.learning_rate)
            criterion = torch.nn.NLLLoss(ignore_index=train_dataset.PAD_idx)
            
            # Training loop
            for epoch in range(config.epochs):
                # Train
                train_loss = train(model, train_loader, optimizer, criterion, config.clip, device)
                
                # Evaluate
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                
                # Log to wandb
                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'epoch': epoch
                })
            
            wandb.finish()
    
    # Run sweep
    wandb.agent(sweep_id, function=sweep_train, count=10)  # Adjust count as needed

if __name__ == "__main__":
    run_vanilla_sweep()
