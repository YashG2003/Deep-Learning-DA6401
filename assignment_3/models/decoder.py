import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers, dropout=0.2, cell_type='lstm'):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        
        # Embedding layer
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        
        # RNN layer
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        else:  # Default to RNN
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, x, hidden):
        # x shape: [batch_size, 1]
        
        # Embedding
        embedded = self.dropout(self.embedding(x))
        # embedded shape: [batch_size, 1, embedding_size]
        
        # RNN forward pass
        if self.cell_type == 'lstm':
            output, (hidden, cell) = self.rnn(embedded, hidden)
            # output shape: [batch_size, 1, hidden_size]
            
            prediction = self.softmax(self.fc_out(output))
            # prediction shape: [batch_size, 1, output_size]
            
            return prediction, (hidden, cell)
        else:  # GRU or RNN
            output, hidden = self.rnn(embedded, hidden)
            # output shape: [batch_size, 1, hidden_size]
            
            prediction = self.softmax(self.fc_out(output))
            # prediction shape: [batch_size, 1, output_size]
            
            return prediction, hidden

class AttentionDecoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers, dropout=0.2, cell_type='lstm'):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        
        # Embedding layer
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        
        # Import here to avoid circular imports
        from models.attention import Attention
        
        # Attention layer
        self.attention = Attention(hidden_size)
        
        # RNN layer
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(embedding_size + hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(embedding_size + hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        else:  # Default to RNN
            self.rnn = nn.RNN(embedding_size + hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size * 2, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, x, hidden, encoder_outputs):
        # x shape: [batch_size, 1]
        # hidden: [n_layers, batch_size, hidden_size]
        # encoder_outputs: [batch_size, src_len, hidden_size]
        
        # Embedding
        embedded = self.dropout(self.embedding(x))
        # embedded shape: [batch_size, 1, embedding_size]
        
        # Calculate attention weights
        attn_weights = self.attention(hidden, encoder_outputs)
        # attn_weights: [batch_size, src_len]
        
        # Apply attention to encoder outputs
        attn_weights = attn_weights.unsqueeze(1)
        # attn_weights: [batch_size, 1, src_len]
        
        context = torch.bmm(attn_weights, encoder_outputs)
        # context: [batch_size, 1, hidden_size]
        
        # Combine embedded input and context vector
        rnn_input = torch.cat((embedded, context), dim=2)
        # rnn_input: [batch_size, 1, embedding_size + hidden_size]
        
        # RNN forward pass
        if self.cell_type == 'lstm':
            output, (hidden, cell) = self.rnn(rnn_input, hidden)
            # output shape: [batch_size, 1, hidden_size]
            
            # Combine output and context for prediction
            output = torch.cat((output, context), dim=2)
            # output: [batch_size, 1, hidden_size * 2]
            
            prediction = self.softmax(self.fc_out(output))
            # prediction shape: [batch_size, 1, output_size]
            
            return prediction, (hidden, cell)
        else:  # GRU or RNN
            output, hidden = self.rnn(rnn_input, hidden)
            # output shape: [batch_size, 1, hidden_size]
            
            # Combine output and context for prediction
            output = torch.cat((output, context), dim=2)
            # output: [batch_size, 1, hidden_size * 2]
            
            prediction = self.softmax(self.fc_out(output))
            # prediction shape: [batch_size, 1, output_size]
            
            return prediction, hidden
