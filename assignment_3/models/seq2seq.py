import torch
import torch.nn as nn
import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        # source shape: [batch_size, source_len]
        # target shape: [batch_size, target_len]
        
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.output_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        
        # Encode the source sequence
        _, encoder_hidden = self.encoder(source)
        
        # First input to the decoder is the <sos> token
        decoder_input = target[:, 0].unsqueeze(1)
        
        # Pass the encoder hidden state to the decoder
        decoder_hidden = self.adjust_hidden_state(encoder_hidden)
        
        # Teacher forcing: use real target outputs as next input with probability
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
        # Decode one step at a time
        for t in range(1, target_len):
            # Pass decoder input and hidden state to decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
            # Store prediction
            outputs[:, t, :] = decoder_output.squeeze(1)
            
            # Get the highest predicted token
            top1 = decoder_output.argmax(2)
            
            # Teacher forcing
            decoder_input = target[:, t].unsqueeze(1) if use_teacher_forcing else top1
        
        return outputs
    
    def adjust_hidden_state(self, encoder_hidden):
        """Adjust encoder hidden state to match decoder layers"""
        # For LSTM, encoder_hidden is a tuple (hidden, cell)
        if isinstance(encoder_hidden, tuple):
            encoder_h, encoder_c = encoder_hidden
            encoder_layers = encoder_h.size(0)
            decoder_layers = self.decoder.num_layers
            
            if encoder_layers == decoder_layers:
                return encoder_hidden
            
            # If encoder has fewer layers than decoder, repeat the last layer
            elif encoder_layers < decoder_layers:
                new_h = encoder_h.clone()
                new_c = encoder_c.clone()
                
                # Pad with copies of the last layer
                for i in range(decoder_layers - encoder_layers):
                    new_h = torch.cat([new_h, encoder_h[-1:]], dim=0)
                    new_c = torch.cat([new_c, encoder_c[-1:]], dim=0)
                
                return (new_h, new_c)
            
            # If encoder has more layers than decoder, take the last n layers
            else:
                return (encoder_h[-decoder_layers:], encoder_c[-decoder_layers:])
        
        # For GRU/RNN, encoder_hidden is just the hidden state
        else:
            encoder_layers = encoder_hidden.size(0)
            decoder_layers = self.decoder.num_layers
            
            if encoder_layers == decoder_layers:
                return encoder_hidden
            
            # If encoder has fewer layers than decoder, repeat the last layer
            elif encoder_layers < decoder_layers:
                new_h = encoder_hidden.clone()
                
                # Pad with copies of the last layer
                for i in range(decoder_layers - encoder_layers):
                    new_h = torch.cat([new_h, encoder_hidden[-1:]], dim=0)
                
                return new_h
            
            # If encoder has more layers than decoder, take the last n layers
            else:
                return encoder_hidden[-decoder_layers:]
    
    def beam_search_decode(self, source, target_vocab, beam_width=3, max_length=50):
        # Encode the source sequence
        _, encoder_hidden = self.encoder(source)
        
        # Adjust hidden state to match decoder layers
        decoder_hidden = self.adjust_hidden_state(encoder_hidden)
        
        # Start with <SOW> token
        start_token = torch.tensor([[target_vocab[0]['>']]]).to(self.device)
        
        # Initialize beams: (sequence, score, hidden_state)
        beams = [(start_token, 0.0, decoder_hidden)]
        completed_beams = []
        
        # Beam search loop
        for _ in range(max_length):
            new_beams = []
            
            for beam_seq, beam_score, beam_hidden in beams:
                # If the last token is <EOW>, add to completed beams
                if beam_seq[0, -1].item() == target_vocab[0]['<']:
                    completed_beams.append((beam_seq, beam_score))
                    continue
                
                # Decode one step
                decoder_output, decoder_hidden = self.decoder(beam_seq[:, -1:], beam_hidden)
                
                # Get top-k predictions
                log_probs, indices = decoder_output[0].topk(beam_width)
                
                # Create new beams
                for i in range(beam_width):
                    token_idx = indices[0, i].item()
                    token_prob = log_probs[0, i].item()
                    
                    # Create new sequence
                    new_seq = torch.cat([beam_seq, torch.tensor([[token_idx]]).to(self.device)], dim=1)
                    
                    # Update score (sum of log probabilities)
                    new_score = beam_score + token_prob
                    
                    # Add to new beams
                    new_beams.append((new_seq, new_score, decoder_hidden))
            
            # Keep only the top beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # If all beams are completed, break
            if not beams:
                break
        
        # If no completed beams, take the best incomplete one
        if not completed_beams and beams:
            completed_beams = [(beams[0][0], beams[0][1])]
        
        # Return the best beam
        if completed_beams:
            best_beam = max(completed_beams, key=lambda x: x[1])
            return best_beam[0][0].cpu().numpy()
        else:
            return []

class AttentionSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(AttentionSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        # source shape: [batch_size, source_len]
        # target shape: [batch_size, target_len]
        
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.output_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        
        # Encode the source sequence
        encoder_outputs, encoder_hidden = self.encoder(source)
        
        # First input to the decoder is the <sos> token
        decoder_input = target[:, 0].unsqueeze(1)
        
        # Adjust hidden state to match decoder layers
        decoder_hidden = self.adjust_hidden_state(encoder_hidden)
        
        # Teacher forcing: use real target outputs as next input with probability
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
        # Decode one step at a time
        for t in range(1, target_len):
            # Pass decoder input, hidden state, and encoder outputs to decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            
            # Store prediction
            outputs[:, t, :] = decoder_output.squeeze(1)
            
            # Get the highest predicted token
            top1 = decoder_output.argmax(2)
            
            # Teacher forcing
            decoder_input = target[:, t].unsqueeze(1) if use_teacher_forcing else top1
        
        return outputs
    
    def adjust_hidden_state(self, encoder_hidden):
        """Adjust encoder hidden state to match decoder layers"""
        # For LSTM, encoder_hidden is a tuple (hidden, cell)
        if isinstance(encoder_hidden, tuple):
            encoder_h, encoder_c = encoder_hidden
            encoder_layers = encoder_h.size(0)
            decoder_layers = self.decoder.num_layers
            
            if encoder_layers == decoder_layers:
                return encoder_hidden
            
            # If encoder has fewer layers than decoder, repeat the last layer
            elif encoder_layers < decoder_layers:
                new_h = encoder_h.clone()
                new_c = encoder_c.clone()
                
                # Pad with copies of the last layer
                for i in range(decoder_layers - encoder_layers):
                    new_h = torch.cat([new_h, encoder_h[-1:]], dim=0)
                    new_c = torch.cat([new_c, encoder_c[-1:]], dim=0)
                
                return (new_h, new_c)
            
            # If encoder has more layers than decoder, take the last n layers
            else:
                return (encoder_h[-decoder_layers:], encoder_c[-decoder_layers:])
        
        # For GRU/RNN, encoder_hidden is just the hidden state
        else:
            encoder_layers = encoder_hidden.size(0)
            decoder_layers = self.decoder.num_layers
            
            if encoder_layers == decoder_layers:
                return encoder_hidden
            
            # If encoder has fewer layers than decoder, repeat the last layer
            elif encoder_layers < decoder_layers:
                new_h = encoder_hidden.clone()
                
                # Pad with copies of the last layer
                for i in range(decoder_layers - encoder_layers):
                    new_h = torch.cat([new_h, encoder_hidden[-1:]], dim=0)
                
                return new_h
            
            # If encoder has more layers than decoder, take the last n layers
            else:
                return encoder_hidden[-decoder_layers:]
    
    def beam_search_decode(self, source, target_vocab, beam_width=3, max_length=50):
        # Encode the source sequence
        encoder_outputs, encoder_hidden = self.encoder(source)
        
        # Adjust hidden state to match decoder layers
        decoder_hidden = self.adjust_hidden_state(encoder_hidden)
        
        # Start with <SOW> token
        start_token = torch.tensor([[target_vocab[0]['>']]]).to(self.device)
        
        # Initialize beams: (sequence, score, hidden_state)
        beams = [(start_token, 0.0, decoder_hidden)]
        completed_beams = []
        
        # Beam search loop
        for _ in range(max_length):
            new_beams = []
            
            for beam_seq, beam_score, beam_hidden in beams:
                # If the last token is <EOW>, add to completed beams
                if beam_seq[0, -1].item() == target_vocab[0]['<']:
                    completed_beams.append((beam_seq, beam_score))
                    continue
                
                # Decode one step
                decoder_output, decoder_hidden = self.decoder(beam_seq[:, -1:], beam_hidden, encoder_outputs)
                
                # Get top-k predictions
                log_probs, indices = decoder_output[0].topk(beam_width)
                
                # Create new beams
                for i in range(beam_width):
                    token_idx = indices[0, i].item()
                    token_prob = log_probs[0, i].item()
                    
                    # Create new sequence
                    new_seq = torch.cat([beam_seq, torch.tensor([[token_idx]]).to(self.device)], dim=1)
                    
                    # Update score (sum of log probabilities)
                    new_score = beam_score + token_prob
                    
                    # Add to new beams
                    new_beams.append((new_seq, new_score, decoder_hidden))
            
            # Keep only the top beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # If all beams are completed, break
            if not beams:
                break
        
        # If no completed beams, take the best incomplete one
        if not completed_beams and beams:
            completed_beams = [(beams[0][0], beams[0][1])]
        
        # Return the best beam
        if completed_beams:
            best_beam = max(completed_beams, key=lambda x: x[1])
            return best_beam[0][0].cpu().numpy()
        else:
            return []
