import argparse
import torch
import os
import wandb

def main():
    parser = argparse.ArgumentParser(description='Dakshina Transliteration')
    parser.add_argument('--mode', type=str, default='train_vanilla', 
                        choices=['train_vanilla', 'train_attention', 'predict', 'visualize', 'sweep_vanilla', 'sweep_attention'],
                        help='Mode to run the script in')
    parser.add_argument('--data_dir', type=str, default='data/dakshina_dataset_v1.0/hi/lexicons/',
                        help='Directory containing the data files')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model for prediction or visualization')
    parser.add_argument('--beam_width', type=int, default=3,
                        help='Beam width for beam search decoding')
    parser.add_argument('--visualize_type', type=str, default='attention', choices=['attention', 'neuron'],
                        help='Type of visualization to generate')
    parser.add_argument('--neuron_idx', type=int, default=None,
                        help='Neuron index to visualize (for neuron visualization)')
    args = parser.parse_args()

    if args.mode == 'train_vanilla':
        from train_vanilla import train_vanilla_model
        train_vanilla_model(args.data_dir)
    elif args.mode == 'train_attention':
        from train_attention import train_attention_model
        train_attention_model(args.data_dir)
    elif args.mode == 'predict':
        from predict import predict
        if args.model_path is None:
            print("Error: Model path must be provided for prediction mode")
            return
        predict(args.model_path, args.data_dir, args.beam_width)
    elif args.mode == 'visualize':
        if args.model_path is None:
            print("Error: Model path must be provided for visualization mode")
            return
        
        if args.visualize_type == 'attention':
            from utils.visualization import generate_attention_heatmaps
            from data.dataset import TransliterationDataset
            from models.encoder import Encoder
            from models.decoder import AttentionDecoder
            from models.seq2seq import AttentionSeq2Seq
            
            # Load dataset
            train_dataset = TransliterationDataset(os.path.join(args.data_dir, 'hi.translit.sampled.train.tsv'), build_vocab=True)
            test_dataset = TransliterationDataset(os.path.join(args.data_dir, 'hi.translit.sampled.test.tsv'), 
                                                 train_dataset.get_vocab()[0], train_dataset.get_vocab()[1])
            
            # Create model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            source_vocab, target_vocab = train_dataset.get_vocab()
            
            encoder = Encoder(
                input_size=len(source_vocab[0]),
                embedding_size=128,
                hidden_size=512,
                num_layers=3,
                dropout=0.3,
                cell_type='gru'
            )
            
            decoder = AttentionDecoder(
                output_size=len(target_vocab[0]),
                embedding_size=128,
                hidden_size=512,
                num_layers=3,
                dropout=0.3,
                cell_type='gru'
            )
            
            model = AttentionSeq2Seq(encoder, decoder, device).to(device)
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            
            generate_attention_heatmaps(model, test_dataset, args.model_path)
        
        elif args.visualize_type == 'neuron':
            from utils.visualization import visualize_neuron_activations
            from data.dataset import TransliterationDataset
            
            if args.neuron_idx is None:
                print("Error: Neuron index must be provided for neuron visualization")
                return
            
            # Load dataset
            train_dataset = TransliterationDataset(os.path.join(args.data_dir, 'hi.translit.sampled.train.tsv'), build_vocab=True)
            test_dataset = TransliterationDataset(os.path.join(args.data_dir, 'hi.translit.sampled.test.tsv'), 
                                                 train_dataset.get_vocab()[0], train_dataset.get_vocab()[1])
            
            visualize_neuron_activations(None, test_dataset, args.model_path, args.neuron_idx)
    elif args.mode == 'sweep_vanilla':
        from sweeps.vanilla_sweep import run_vanilla_sweep
        run_vanilla_sweep()
    elif args.mode == 'sweep_attention':
        from sweeps.attention_sweep import run_attention_sweep
        run_attention_sweep()

if __name__ == '__main__':
    main()
