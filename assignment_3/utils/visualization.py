import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import urllib.request
from matplotlib import font_manager

def setup_devanagari_font():
    """
    Download and setup Devanagari font for matplotlib
    """
    font_path = 'NotoSansDevanagari-Regular.ttf'
    
    if not os.path.exists(font_path):
        # Download Noto Sans Devanagari font
        urllib.request.urlretrieve(
            'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf',
            font_path
        )
    
    # Register the font with matplotlib
    font_manager.fontManager.addfont(font_path)
    font_prop = font_manager.FontProperties(fname=font_path)
    
    return font_prop

def generate_attention_heatmaps(model, test_dataset, model_path, num_examples=9, beam_width=3):
    """
    Generate attention heatmaps for test examples
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    font_prop = setup_devanagari_font()
    
    # Create directory for attention heatmaps
    os.makedirs('attention_heatmaps', exist_ok=True)
    
    # Create a modified attention decoder that saves attention weights
    from models.decoder import AttentionDecoder
    
    class AttentionDecoderWithSave(AttentionDecoder):
        def __init__(self, *args, **kwargs):
            super(AttentionDecoderWithSave, self).__init__(*args, **kwargs)
            self.saved_attention = []
        
        def forward(self, x, hidden, encoder_outputs):
            # Embedding
            embedded = self.dropout(self.embedding(x))
            
            # Calculate attention weights
            attn_weights = self.attention(hidden, encoder_outputs)
            
            # Save attention weights for visualization
            self.saved_attention.append(attn_weights.detach().cpu())
            
            # Apply attention to encoder outputs
            attn_weights = attn_weights.unsqueeze(1)
            context = torch.bmm(attn_weights, encoder_outputs)
            
            # Combine embedded input and context vector
            rnn_input = torch.cat((embedded, context), dim=2)
            
            # RNN forward pass
            if self.cell_type == 'lstm':
                output, (hidden, cell) = self.rnn(rnn_input, hidden)
                output = torch.cat((output, context), dim=2)
                prediction = self.softmax(self.fc_out(output))
                return prediction, (hidden, cell)
            else:  # GRU or RNN
                output, hidden = self.rnn(rnn_input, hidden)
                output = torch.cat((output, context), dim=2)
                prediction = self.softmax(self.fc_out(output))
                return prediction, hidden
    
    # Load model
    from models.encoder import Encoder
    from models.seq2seq import AttentionSeq2Seq
    
    # Get vocabulary
    source_vocab, target_vocab = test_dataset.get_vocab()
    
    # Create model with same architecture as training
    encoder = Encoder(
        input_size=len(source_vocab[0]),
        embedding_size=128,
        hidden_size=512,
        num_layers=3,
        dropout=0.3,
        cell_type='gru'
    )
    
    # Create the decoder with attention saving capability
    decoder = AttentionDecoderWithSave(
        output_size=len(target_vocab[0]),
        embedding_size=128,
        hidden_size=512,
        num_layers=3,
        dropout=0.3,
        cell_type='gru'
    )
    
    # Create the seq2seq model
    model = AttentionSeq2Seq(encoder, decoder, device).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Select random examples from test set
    random.seed(42)  # For reproducibility
    example_indices = random.sample(range(len(test_dataset)), num_examples)
    
    # Generate attention heatmaps
    examples = []
    
    with torch.no_grad():
        for idx in example_indices:
            item = test_dataset[idx]
            source = item['source'].unsqueeze(0).to(device)  # Add batch dimension
            source_text = item['source_text']
            target_text = item['target_text']
            
            # Clear saved attention
            model.decoder.saved_attention = []
            
            # Generate prediction
            encoder_outputs, encoder_hidden = model.encoder(source)
            decoder_hidden = model.adjust_hidden_state(encoder_hidden)
            
            # Start with <SOW> token
            decoder_input = torch.tensor([[test_dataset.SOW_idx]]).to(device)
            
            # Store predicted indices and attention weights
            predicted_indices = []
            
            # Generate characters one by one
            for _ in range(50):  # Maximum length
                decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
                
                # Get the highest probability character
                top1 = decoder_output.argmax(2)
                predicted_indices.append(top1.item())
                
                # If end token is predicted, stop
                if top1.item() == test_dataset.EOW_idx:
                    break
                
                # Use predicted token as next input
                decoder_input = top1
            
            # Convert indices to characters
            pred_word = []
            for idx in predicted_indices:
                if idx == test_dataset.EOW_idx:
                    break
                if idx > 3:  # Skip special tokens
                    pred_word.append(target_vocab[1][idx])
            
            pred_word = ''.join(pred_word)
            
            # Get attention weights
            attention_weights = torch.cat(model.decoder.saved_attention, dim=0).numpy()
            
            # Store example data
            examples.append({
                'source': source_text,
                'target': target_text,
                'prediction': pred_word,
                'correct': pred_word == target_text,
                'attention': attention_weights
            })
    
    # Plot attention heatmaps in a grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, example in enumerate(examples):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Get source and prediction characters
        source_chars = list(example['source'])
        pred_chars = list(example['prediction'])
        
        # Get attention weights (trim to actual length)
        attention = example['attention'][:len(pred_chars), :len(source_chars)]
        
        # Plot heatmap
        im = ax.imshow(attention, cmap='YlGnBu')
        
        # Set labels with proper font
        ax.set_xticks(np.arange(len(source_chars)))
        ax.set_yticks(np.arange(len(pred_chars)))
        ax.set_xticklabels(source_chars)
        ax.set_yticklabels(pred_chars, fontproperties=font_prop)
        
        # Remove title to avoid rendering issues
        ax.set_title("")
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Hide any unused subplots
    for i in range(len(examples), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('attention_heatmaps/grid.png', dpi=300)
    plt.show()
    
    return examples

def visualize_neuron_activations(model, test_dataset, model_path, neuron_idx, num_examples=15):
    """
    Visualize activations of a specific neuron for multiple examples
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    font_prop = setup_devanagari_font()
    
    # Create directory for visualizations
    os.makedirs('neuron_visualizations', exist_ok=True)
    
    # Create a modified decoder that saves hidden states
    from models.decoder import AttentionDecoder
    
    class DecoderWithStateCapture(AttentionDecoder):
        def __init__(self, *args, **kwargs):
            super(DecoderWithStateCapture, self).__init__(*args, **kwargs)
            self.saved_states = []
        
        def forward(self, x, hidden, encoder_outputs):
            # Original implementation
            embedded = self.dropout(self.embedding(x))
            attn_weights = self.attention(hidden, encoder_outputs)
            
            # Save attention weights for visualization
            attn_weights = attn_weights.unsqueeze(1)
            context = torch.bmm(attn_weights, encoder_outputs)
            rnn_input = torch.cat((embedded, context), dim=2)
            
            if self.cell_type == 'lstm':
                output, (hidden, cell) = self.rnn(rnn_input, hidden)
                # Save hidden state for visualization
                self.saved_states.append(hidden[-1].detach().cpu())
                
                output = torch.cat((output, context), dim=2)
                prediction = self.softmax(self.fc_out(output))
                return prediction, (hidden, cell)
            else:  # GRU or RNN
                output, hidden = self.rnn(rnn_input, hidden)
                # Save hidden state for visualization
                self.saved_states.append(hidden[-1].detach().cpu())
                
                output = torch.cat((output, context), dim=2)
                prediction = self.softmax(self.fc_out(output))
                return prediction, hidden
    
    # Load model
    from models.encoder import Encoder
    from models.seq2seq import AttentionSeq2Seq
    
    # Get vocabulary
    source_vocab, target_vocab = test_dataset.get_vocab()
    
    # Create model with same architecture as training
    encoder = Encoder(
        input_size=len(source_vocab[0]),
        embedding_size=128,
        hidden_size=512,
        num_layers=3,
        dropout=0.3,
        cell_type='gru'
    )
    
    # Create the decoder with state capture
    decoder = DecoderWithStateCapture(
        output_size=len(target_vocab[0]),
        embedding_size=128,
        hidden_size=512,
        num_layers=3,
        dropout=0.3,
        cell_type='gru'
    )
    
    # Create the seq2seq model
    model = AttentionSeq2Seq(encoder, decoder, device).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Select random examples from test set
    random.seed(42)  # For reproducibility
    example_indices = random.sample(range(len(test_dataset)), num_examples)
    
    # Generate predictions and collect activations
    examples = []
    
    with torch.no_grad():
        for idx in example_indices:
            item = test_dataset[idx]
            source = item['source'].unsqueeze(0).to(device)  # Add batch dimension
            source_text = item['source_text']
            target_text = item['target_text']
            
            # Clear saved states
            model.decoder.saved_states = []
            
            # Generate prediction
            encoder_outputs, encoder_hidden = model.encoder(source)
            decoder_hidden = model.adjust_hidden_state(encoder_hidden)
            
            # Start with <SOW> token
            decoder_input = torch.tensor([[test_dataset.SOW_idx]]).to(device)
            
            # Store predicted indices
            predicted_indices = []
            
            # Generate characters one by one
            for _ in range(50):  # Maximum length
                decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
                
                # Get the highest probability character
                top1 = decoder_output.argmax(2)
                predicted_indices.append(top1.item())
                
                # If end token is predicted, stop
                if top1.item() == test_dataset.EOW_idx:
                    break
                
                # Use predicted token as next input
                decoder_input = top1
            
            # Convert indices to characters
            pred_word = []
            for idx in predicted_indices:
                if idx == test_dataset.EOW_idx:
                    break
                if idx > 3:  # Skip special tokens
                    pred_word.append(target_vocab[1][idx])
            
            pred_word = ''.join(pred_word)
            
            # Get hidden states
            hidden_states = torch.stack(model.decoder.saved_states)
            
            # Apply sigmoid to get activation probabilities
            activations = torch.sigmoid(hidden_states).numpy()
            
            # Extract activations for the specific neuron
            neuron_activations = activations[:, 0, neuron_idx]
            
            # Store example data
            examples.append({
                'source': source_text,
                'target': target_text,
                'prediction': pred_word,
                'correct': pred_word == target_text,
                'activations': neuron_activations
            })
    
    # Function to get color based on activation value
    def get_color(value):
        """
        Returns a color between blue and red based on value (0-1)
        Blue (0) -> White (0.5) -> Red (1)
        """
        if value < 0.5:
            # Blue (0) to White (0.5)
            r = value * 2
