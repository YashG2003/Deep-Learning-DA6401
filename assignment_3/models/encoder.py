import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout=0.2, cell_type='lstm'):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        
        # Embedding layer
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        
        # RNN layer
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        else:  # Default to RNN
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        
        # Embedding
        embedded = self.dropout(self.embedding(x))
        # embedded shape: [batch_size, seq_len, embedding_size]
        
        # RNN forward pass
        if self.cell_type == 'lstm':
            outputs, (hidden, cell) = self.rnn(embedded)
            # outputs shape: [batch_size, seq_len, hidden_size]
            # hidden shape: [n_layers, batch_size, hidden_size]
            # cell shape: [n_layers, batch_size, hidden_size]
            
            return outputs, (hidden, cell)
        else:  # GRU or RNN
            outputs, hidden = self.rnn(embedded)
            # outputs shape: [batch_size, seq_len, hidden_size]
            # hidden shape: [n_layers, batch_size, hidden_size]
            
            return outputs, hidden
