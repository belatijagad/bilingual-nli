# training/trainer.py
import torch
import torch.distributed as dist
from tqdm import tqdm

def train_model_epoch(config, model, train_dataloader, optimizers_list, loss_fn, device, epoch, rank, world_size, is_main_process, train_sampler):
    model.train()
    if train_sampler: 
        train_sampler.set_epoch(epoch)
        
    epoch_loss = 0.0
    
    batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_train_epochs']}", 
                          disable=not is_main_process, position=0, leave=True)
    
    for batch_idx, batch in enumerate(batch_iterator):
        input_ids, attention_mask, labels = [b.to(device, non_blocking=True) for b in batch]
        
        for optimizer in optimizers_list:
            optimizer.zero_grad(set_to_none=True)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        
        loss.backward()
        
        for optimizer in optimizers_list:
            optimizer.step()
        
        epoch_loss += loss.item()
        if is_main_process:
            batch_iterator.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}'})
            
    avg_epoch_loss = epoch_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
    
    if world_size > 1:
        loss_tensor = torch.tensor([avg_epoch_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_epoch_loss = loss_tensor.item()
    
    return avg_epoch_loss