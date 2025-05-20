import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [n_layers, batch_size, hidden_size] or tuple of tensors for LSTM
        # encoder_outputs: [batch_size, src_len, hidden_size]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Get the last layer hidden state
        if isinstance(hidden, tuple):  # LSTM case
            hidden = hidden[0]  # Use only the hidden state, not the cell state
        
        # Take the last layer's hidden state
        hidden = hidden[-1]  # Shape: [batch_size, hidden_size]
        
        # Reshape hidden to [batch_size, 1, hidden_size] and repeat for each source token
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Calculate energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy: [batch_size, src_len, hidden_size]
        
        # Calculate attention weights
        attention = self.v(energy).squeeze(2)
        # attention: [batch_size, src_len]
        
        return F.softmax(attention, dim=1)
