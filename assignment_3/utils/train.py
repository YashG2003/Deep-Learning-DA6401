import torch
import time
import wandb

def train(model, iterator, optimizer, criterion, clip, device, teacher_forcing_ratio=0.5):
    model.train()
    epoch_loss = 0
    
    for batch in iterator:
        # Get source and target tensors
        source = batch['source'].to(device)
        target = batch['decoder_input'].to(device)
        target_output = batch['decoder_target'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(source, target, teacher_forcing_ratio)
        
        # Calculate loss (ignore padding)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)  # Remove first token (<SOW>)
        target_output = target_output[:, :-1].reshape(-1)  # Remove last token (padding)
        
        loss = criterion(output, target_output)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update parameters
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def train_model(model, train_loader, val_loader, optimizer, criterion, config, device):
    """
    Train a model with the given configuration
    """
    # Training loop
    best_val_acc = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    val_accs = []
    
    from utils.evaluate import evaluate
    
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        # Train
        train_loss = train(model, train_loader, optimizer, criterion, config['clip'], device)
        train_losses.append(train_loss)
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs:.2f}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tVal. Loss: {val_loss:.3f} | Val. Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            torch.save(model.state_dict(), f'{config["model_name"]}.pt')
        
        # Log to wandb
        if config['use_wandb']:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'epoch': epoch
            })
    
    return best_model_state, train_losses, val_losses, val_accs
