import os
import torch
import torch.distributed as dist

def initialize_ddp_environment():
    if not dist.is_available() or not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        print("DDP not available or no CUDA GPUs found. Running in single CPU mode.")
        return 0, 1, 0, torch.device("cpu")

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else: 
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
        if device.type == "cuda": torch.cuda.set_device(device)
            
    return rank, world_size, local_rank, device

def cleanup_ddp(world_size):
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()