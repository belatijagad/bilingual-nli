import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.config_loader import load_app_config
from utils.distributed_setup import initialize_ddp_environment, cleanup_ddp
from utils.data_processing import load_and_prepare_data
from utils.model_setup import get_tokenizer, get_model, wrap_model_ddp
from training.optimizers import setup_optimizers
from training.trainer import train_model_epoch
from inference.predictor import run_inference_engine

def main(cli_args):
    config = load_app_config(cli_args.config_file)
    rank, world_size, local_rank, device = initialize_ddp_environment()
    is_main_process = (rank == 0)

    if is_main_process:
        print(f"--- Mode: {cli_args.mode.upper()} ---")
        print("--- Application Configuration ---")
        for section_key, section_value in config.items():
            if isinstance(section_value, dict) and section_key in ["paths", "model", "training", "flags"]:
                 print(f"  Section [{section_key}]:")
                 for key, value in section_value.items(): print(f"    {key}: {value}")
            elif not isinstance(section_value, dict):
                 print(f"  {section_key}: {section_value}")
        print("--- DDP Setup ---")
        print(f"Rank: {rank}, World Size: {world_size}, Local Rank (GPU Index): {local_rank}, Device: {device}")
        print("-" * 30)

    # --- Training Mode ---
    if cli_args.mode == 'train':
        if not config["perform_training"]:
            if is_main_process: print("Training mode selected, but 'perform_training' is false in config. Exiting.")
            cleanup_ddp(world_size)
            return

        tokenizer = get_tokenizer(config)
        # For training, we primarily need the train_dataset
        train_dataset, _, _ = load_and_prepare_data(config, tokenizer) 
        
        if train_dataset is None:
            if is_main_process: print("Error: Training dataset is None or could not be created. Cannot train.")
            cleanup_ddp(world_size)
            return

        model_instance = get_model(config)
        # DDP wrapping and moving to device is handled by wrap_model_ddp
        ddp_model = wrap_model_ddp(model_instance, device, world_size) 
        
        if is_main_process: print(f"\n--- Starting Training for {config['num_train_epochs']} epochs ---")
        
        model_for_optimizer_setup = ddp_model.module if world_size > 1 else ddp_model
        optimizers_list = setup_optimizers(config, model_for_optimizer_setup, rank, world_size)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        train_sampler = None
        if world_size > 1:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        
        num_workers_per_process = 0
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            if world_size > 0 :
                 num_workers_per_process = max(1, (cpu_count // world_size) // 2 )
            else: # Should not happen if world_size is properly set
                 num_workers_per_process = max(1, cpu_count // 2)
        else:
            num_workers_per_process = 2 # Default if cpu_count is None

        train_dataloader = DataLoader(
            train_dataset, batch_size=config["train_batch_size_per_gpu"],
            sampler=train_sampler, pin_memory=True, 
            shuffle=(train_sampler is None), num_workers=num_workers_per_process
        )

        for epoch in range(config["num_train_epochs"]):
            avg_loss = train_model_epoch(config, ddp_model, train_dataloader, optimizers_list, loss_fn, device, epoch, rank, world_size, is_main_process, train_sampler)
            if is_main_process:
                print(f"Epoch {epoch+1} Avg Training Loss: {avg_loss:.4f}")
        
        if is_main_process: print("--- Training Finished ---")

        if is_main_process and config["trained_model_output_dir"]:
            print(f"Saving model to {config['trained_model_output_dir']}")
            model_to_save = ddp_model.module if world_size > 1 else ddp_model
            model_to_save.save_pretrained(config['trained_model_output_dir'])
            tokenizer.save_pretrained(config['trained_model_output_dir'])
            print("Model and tokenizer saved.")
    
    # --- Prediction Mode ---
    elif cli_args.mode == 'predict':
        if not config["perform_inference"]:
            if is_main_process: print("Prediction mode selected, but 'perform_inference' is false in config. Exiting.")
            cleanup_ddp(world_size)
            return
            
        model_path_to_load = cli_args.model_path_for_predict if cli_args.model_path_for_predict else config["trained_model_output_dir"]
        
        # Check if model files actually exist at the path
        if not (os.path.exists(os.path.join(model_path_to_load, "pytorch_model.bin")) or \
                os.path.exists(os.path.join(model_path_to_load, "model.safetensors"))):
            if is_main_process:
                print(f"Error: Model not found at '{model_path_to_load}'. Evaluated paths:")
                print(f"  - {os.path.join(model_path_to_load, 'pytorch_model.bin')}")
                print(f"  - {os.path.join(model_path_to_load, 'model.safetensors')}")
                print("Please train a model first or provide a valid --model_path_for_predict argument.")
            cleanup_ddp(world_size)
            return

        if is_main_process: print(f"Loading model and tokenizer from {model_path_to_load} for prediction.")
        
        # Create a temporary config for loading model and tokenizer from the specified path
        # This ensures get_model and get_tokenizer use the path as the "model_name"
        predict_tokenizer_config = {"model_name": model_path_to_load, "hf_token": config.get("hf_token")}
        predict_model_config = {"model_name": model_path_to_load, "hf_token": config.get("hf_token")}


        tokenizer = get_tokenizer(predict_tokenizer_config)
        model_for_inference = get_model(predict_model_config)
        model_for_inference.to(device) # Move to the correct device for the (main) inference process
        
        # For prediction, we only need the test_df.
        # We use the original config for data paths, but the (potentially) reloaded tokenizer from model dir.
        _, _, test_df_for_inf = load_and_prepare_data(config, tokenizer) 

        run_inference_engine(config, model_for_inference, tokenizer, test_df_for_inf, device, is_main_process)

    else: # Should not happen due to 'choices' in argparse
        if is_main_process: print(f"Unknown mode: {cli_args.mode}")

    cleanup_ddp(world_size)
    if is_main_process: print(f"Script finished for mode: {cli_args.mode}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLI Training and Inference Script with DDP")
    
    # Positional argument for mode
    parser.add_argument("mode", choices=['train', 'predict'], 
                        help="Operation mode: 'train' to train a new model, or 'predict' to run inference.")
    
    # Optional arguments
    parser.add_argument("--config_file", type=str, default="config.yml", 
                        help="Path to the YAML configuration file (default: config.yml in script dir).")
    parser.add_argument("--model_path_for_predict", type=str, default=None,
                        help="Path to a specific trained model directory for prediction mode. Overrides 'trained_model_output_dir' from config for loading.")
    
    cli_args = parser.parse_args()
    
    # Ensure config_file path is correctly resolved if relative
    if not os.path.isabs(cli_args.config_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cli_args.config_file = os.path.join(script_dir, cli_args.config_file)
        if not os.path.exists(cli_args.config_file):
            print(f"Warning: Default config file '{cli_args.config_file}' not found. Ensure it exists or specify path with --config_file.")
            # Depending on desired behavior, you might exit here or let load_app_config handle it with defaults.
            # For now, load_app_config will print a warning and use defaults.

    main(cli_args)