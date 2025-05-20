import torch
import numpy as np
import pandas as pd
import os

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    correct_words = 0
    total_words = 0
    
    with torch.no_grad():
        for batch in iterator:
            source = batch['source'].to(device)
            target = batch['decoder_input'].to(device)
            target_output = batch['decoder_target'].to(device)
            source_text = batch['source_text']
            target_text = batch['target_text']
            
            # Forward pass
            output = model(source, target, 0)  # No teacher forcing during evaluation
            
            # Calculate loss
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            target_output = target_output[:, :-1].reshape(-1)
            
            loss = criterion(output, target_output)
            
            epoch_loss += loss.item()
            
            # Calculate word accuracy
            batch_size = source.shape[0]
            output = output.reshape(batch_size, -1, output_dim)
            
            for i in range(batch_size):
                # Get predicted word
                pred_indices = output[i].argmax(dim=1)
                pred_word = []
                
                for idx in pred_indices:
                    if idx == 1:  # EOW token
                        break
                    if idx > 3:  # Skip special tokens
                        pred_word.append(iterator.dataset.target_int2char[idx.item()])
                
                pred_word = ''.join(pred_word)
                
                # Check if prediction matches target
                if pred_word == target_text[i]:
                    correct_words += 1
                total_words += 1
    
    return epoch_loss / len(iterator), (correct_words / total_words) * 100

def beam_search_evaluate(model, iterator, device, beam_width=3):
    model.eval()
    correct_words = 0
    total_words = 0
    target_vocab = iterator.dataset.get_vocab()[1]
    
    with torch.no_grad():
        for batch in iterator:
            source = batch['source'].to(device)
            target_text = batch['target_text']
            
            # Process one sample at a time for beam search
            for i in range(source.shape[0]):
                src = source[i:i+1]
                
                # Beam search decoding
                beam_result = model.beam_search_decode(src, target_vocab, beam_width)
                
                # Convert indices to characters
                pred_word = []
                for idx in beam_result:
                    if idx == 1:  # EOW token
                        break
                    if idx > 3:  # Skip special tokens
                        pred_word.append(iterator.dataset.target_int2char[idx])
                
                pred_word = ''.join(pred_word)
                
                # Check if prediction matches target
                if pred_word == target_text[i]:
                    correct_words += 1
                total_words += 1
    
    return (correct_words / total_words) * 100

def generate_predictions(model, test_dataset, device, output_dir, model_name, beam_width=3):
    """
    Generate predictions for all test data and save to a file
    """
    model.eval()
    target_vocab = test_dataset.get_vocab()[1]
    
    # Create directory for predictions
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate predictions
    predictions = []
    
    with torch.no_grad():
        for i, item in enumerate(test_dataset):
            source = item['source'].unsqueeze(0).to(device)  # Add batch dimension
            source_text = item['source_text']
            target_text = item['target_text']
            
            # Beam search decoding
            beam_result = model.beam_search_decode(source, target_vocab, beam_width)
            
            # Convert indices to characters
            pred_word = []
            for idx in beam_result:
                if idx == test_dataset.EOW_idx:
                    break
                if idx > 3:  # Skip special tokens
                    pred_word.append(target_vocab[1][idx])
            
            pred_word = ''.join(pred_word)
            
            # Store prediction
            predictions.append({
                'source': source_text,
                'target': target_text,
                'prediction': pred_word,
                'correct': pred_word == target_text
            })
    
    # Save predictions to CSV file with UTF-8-BOM encoding for Excel compatibility
    df = pd.DataFrame(predictions)
    df.to_csv(f'{output_dir}/predictions.csv', index=False, encoding='utf-8-sig')
    
    # Also save as text file with UTF-8 encoding
    with open(f'{output_dir}/predictions.txt', 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(f"{pred['source']}\t{pred['prediction']}\t{pred['target']}\t{pred['correct']}\n")
    
    # Calculate accuracy
    correct = sum(1 for pred in predictions if pred['correct'])
    accuracy = correct / len(predictions) * 100
    
    print(f"Generated predictions for {len(predictions)} test examples")
    print(f"Beam Search Accuracy (width={beam_width}): {accuracy:.2f}%")
    
    return predictions
