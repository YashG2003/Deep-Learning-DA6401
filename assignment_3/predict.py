import torch
import os
import pandas as pd
import argparse
from torch.utils.data import DataLoader

from data.dataset import TransliterationDataset
from models.encoder import Encoder
from models.decoder import Decoder, AttentionDecoder
from models.seq2seq import Seq2Seq, AttentionSeq2Seq
from utils.evaluate import generate_predictions

def predict(model_path, data_dir, beam_width=3):
    """
    Load a trained model and generate predictions on the test set
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load train dataset to get vocabulary
    train_dataset = TransliterationDataset(os.path.join(data_dir, 'hi.translit.sampled.train.tsv'), build_vocab=True)
    source_vocab, target_vocab = train_dataset.get_vocab()
    
    # Load test dataset
    test_dataset = TransliterationDataset(os.path.join(data_dir, 'hi.translit.sampled.test.tsv'), 
                                         source_vocab, target_vocab)
    
    # Determine model type from filename
    is_attention = 'attention' in model_path
    output_dir = 'predictions_attention' if is_attention else 'predictions_vanilla'
    model_type = 'attention' if is_attention else 'vanilla'
    
    # Create model with same architecture as training
    source_vocab_size = len(source_vocab[0])
    target_vocab_size = len(target_vocab[0])
    
    # Default configuration (can be adjusted based on the model you're loading)
    if is_attention:
        encoder = Encoder(
            input_size=source_vocab_size,
            embedding_size=128,
            hidden_size=512,
            num_layers=3,
            dropout=0.3,
            cell_type='gru'
        )
        
        decoder = AttentionDecoder(
            output_size=target_vocab_size,
            embedding_size=128,
            hidden_size=512,
            num_layers=3,
            dropout=0.3,
            cell_type='gru'
        )
        
        model = AttentionSeq2Seq(encoder, decoder, device).to(device)
    else:
        encoder = Encoder(
            input_size=source_vocab_size,
            embedding_size=256,
            hidden_size=512,
            num_layers=3,
            dropout=0.2,
            cell_type='lstm'
        )
        
        decoder = Decoder(
            output_size=target_vocab_size,
            embedding_size=256,
            hidden_size=512,
            num_layers=3,
            dropout=0.2,
            cell_type='lstm'
        )
        
        model = Seq2Seq(encoder, decoder, device).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Generate predictions
    predictions = generate_predictions(model, test_dataset, device, output_dir, model_type, beam_width)
    
    return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate predictions using a trained model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, default='data/dakshina_dataset_v1.0/hi/lexicons/',
                        help='Directory containing the data files')
    parser.add_argument('--beam_width', type=int, default=3, help='Beam width for beam search decoding')
    
    args = parser.parse_args()
    predict(args.model_path, args.data_dir, args.beam_width)
