# training/optimizers.py
import torch # Ensure torch is imported if not already
from torch.optim import AdamW

try:
    from muon import Muon 
except ImportError:
    print("Warning: Muon optimizer module not found. Muon optimizer will not be available.")
    Muon = None

def setup_optimizers(config, model_to_configure, rank, world_size):
    """
    Sets up and returns a list of optimizers based on the model and config.
    model_to_configure should be the unwrapped model (e.g., model.module if DDP).
    'rank' is the global rank of the current process.
    """
    is_main_process = (rank == 0) # Define is_main_process using the rank argument

    should_use_muon = config.get("use_muon", False) and Muon is not None

    if config.get("use_muon", False) and Muon is None:
        if is_main_process: # Print critical warnings only on main process to avoid clutter
            print("Critical Warning: Configured to use Muon optimizer (use_muon=true), but Muon library is not loaded/found. Will use AdamW for all parameters instead.")
        should_use_muon = False
    elif Muon is None and (config.get("learning_rate_muon") is not None or config.get("muon_momentum") is not None):
        if is_main_process:
            print("Warning: Muon-specific optimizer parameters (e.g., learning_rate_muon) are present in the config, "
                  "but the Muon library is not loaded. Muon optimizer will not be used.")
        
    if hasattr(model_to_configure, 'module'): # If DDP wrapped
        model_to_configure = model_to_configure.module

    all_trainable_params = [p for p in model_to_configure.parameters() if p.requires_grad]
    muon_specific_params = []
    adamw_specific_params_list_of_lists = [] # Collect groups of params for AdamW

    if should_use_muon:
        if is_main_process: print("Setting up optimizers with Muon (for applicable parameters) and AdamW.")
        # Attempt to split parameters for Muon and AdamW based on DeBERTa structure
        if hasattr(model_to_configure, 'deberta') and hasattr(model_to_configure.deberta, 'encoder'):
            muon_specific_params = [p for p in model_to_configure.deberta.encoder.parameters() if p.ndim >= 2 and p.requires_grad]
            
            current_adamw_group = [p for p in model_to_configure.deberta.encoder.parameters() if p.ndim < 2 and p.requires_grad]
            if hasattr(model_to_configure.deberta, 'embeddings'):
                current_adamw_group.extend(p for p in model_to_configure.deberta.embeddings.parameters() if p.requires_grad)
            
            # Classifier parameters are typically handled by AdamW separately
            if hasattr(model_to_configure, 'classifier'):
                classifier_params_group = [p for p in model_to_configure.classifier.parameters() if p.requires_grad]
                if classifier_params_group: adamw_specific_params_list_of_lists.append(classifier_params_group)

            if current_adamw_group: adamw_specific_params_list_of_lists.append(current_adamw_group)
            
            covered_params = set(muon_specific_params)
            for group in adamw_specific_params_list_of_lists:
                covered_params.update(group)
            
            remaining_params_for_adamw = [p for p in all_trainable_params if p not in covered_params]
            if remaining_params_for_adamw:
                if is_main_process: print(f"Info: {len(remaining_params_for_adamw)} trainable parameters were not explicitly assigned based on DeBERTa structure; adding them to AdamW.")
                adamw_specific_params_list_of_lists.append(remaining_params_for_adamw)
        else: 
            if is_main_process: print("Warning: Model does not have a 'deberta.encoder' attribute or Muon is enabled without specific splitting logic for this model type. Applying heuristic for Muon.")
            muon_specific_params = [p for p in all_trainable_params if p.ndim >= 2][:len(all_trainable_params)//2] 
            adamw_remaining_params_group = [p for p in all_trainable_params if p not in muon_specific_params]
            if adamw_remaining_params_group: adamw_specific_params_list_of_lists.append(adamw_remaining_params_group)
    else: 
        if is_main_process: print("Setting up optimizer with AdamW only for all trainable parameters.")
        if all_trainable_params:
            adamw_specific_params_list_of_lists.append(all_trainable_params)
        muon_specific_params = []

    final_adamw_params = [p for sublist in adamw_specific_params_list_of_lists for p in sublist]
    
    optimizers_list = []
    if should_use_muon and muon_specific_params:
         optimizers_list.append(
             Muon(muon_specific_params, 
                  lr=config.get("learning_rate_muon", 0.01), 
                  momentum=config.get("muon_momentum", 0.9),
                  rank=rank, 
                  world_size=world_size)
        )
    
    if final_adamw_params:
        optimizers_list.append(
            AdamW(final_adamw_params, 
                  lr=config.get("learning_rate_adamw", 2e-5), 
                  betas=(config.get("adamw_betas_0", 0.9), config.get("adamw_betas_1", 0.999)), 
                  weight_decay=config.get("adamw_weight_decay", 0.01))
        )
    
    if not optimizers_list:
        if not all_trainable_params:
            if is_main_process: print("Warning: No trainable parameters found in the model. No optimizers created.")
        else:
            if is_main_process: print("Warning: No optimizers were created despite trainable parameters. Defaulting to AdamW for all parameters.")
            optimizers_list.append(
                AdamW(all_trainable_params,
                      lr=config.get("learning_rate_adamw", 2e-5),
                      betas=(config.get("adamw_betas_0", 0.9), config.get("adamw_betas_1", 0.999)),
                      weight_decay=config.get("adamw_weight_decay", 0.01))
            )
            if not all_trainable_params and not optimizers_list: # Should not be reached if all_trainable_params was empty initially
                 raise ValueError("No trainable parameters found and no optimizers could be created.")

    if is_main_process:
        print(f"Created {len(optimizers_list)} optimizer(s).")
        adamw_present = any(isinstance(opt, AdamW) for opt in optimizers_list)
        muon_present = should_use_muon and any(isinstance(opt, Muon) for opt in optimizers_list)

        if adamw_present:
            print(f"  AdamW LR: {config.get('learning_rate_adamw')}")
        if muon_present: # Check if Muon was intended and actually added
            print(f"  Muon LR: {config.get('learning_rate_muon')}")
            
    return optimizers_list