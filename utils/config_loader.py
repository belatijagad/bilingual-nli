import os
import yaml
from dotenv import load_dotenv

def load_app_config(config_file_path="config.yml"):
    load_dotenv() 

    defaults = {
        "paths": {"train_csv": "train.csv", "test_csv": "test.csv", "trained_model_output_dir": "./model_output", "submission_dir": "./submissions"},
        "model": {"name": "microsoft/deberta-v3-base", "hf_token": None},
        "training": {"max_length": 128, "train_batch_size_per_gpu": 8, "inference_batch_size_per_gpu": 16, "num_epochs": 1,
                     "optimizer": {"adamw": {"lr": 2e-5, "betas": [0.9, 0.999], "weight_decay": 0.01}, "muon": {"lr": 0.01, "momentum": 0.9}}},
        "flags": {"perform_training": False, "perform_inference": False}
    }
    yaml_config = {}
    try:
        with open(config_file_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config is None: yaml_config = {}
    except FileNotFoundError:
        print(f"Warning: Config file '{config_file_path}' not found. Using defaults/env vars.")
    except Exception as e:
        print(f"Error loading config '{config_file_path}': {e}. Using defaults/env vars.")

    def get_val(keys, env_var, default_val_dict, type_cast=str):
        val = os.getenv(env_var)
        if val is not None: return type_cast(val)
        temp_yaml = yaml_config
        for key in keys[:-1]: temp_yaml = temp_yaml.get(key, {})
        val_yaml = temp_yaml.get(keys[-1])
        if val_yaml is not None: return type_cast(val_yaml)
        temp_default = default_val_dict
        for key in keys[:-1]: temp_default = temp_default.get(key, {})
        return type_cast(temp_default.get(keys[-1]))

    config = {}
    config["train_csv_path"] = get_val(['paths', 'train_csv'], "TRAIN_CSV_PATH", defaults)
    config["test_csv_path"] = get_val(['paths', 'test_csv'], "TEST_CSV_PATH", defaults)
    config["trained_model_output_dir"] = get_val(['paths', 'trained_model_output_dir'], "TRAINED_MODEL_OUTPUT_DIR", defaults)
    config["submission_dir"] = get_val(['paths', 'submission_dir'], "SUBMISSION_DIR", defaults)
    config["model_name"] = get_val(['model', 'name'], "MODEL_NAME", defaults)
    config["hf_token"] = get_val(['model', 'hf_token'], "HF_TOKEN", defaults, lambda x: x if x else None)
    config["max_length"] = get_val(['training', 'max_length'], "MAX_LENGTH", defaults, int)
    config["train_batch_size_per_gpu"] = get_val(['training', 'train_batch_size_per_gpu'], "TRAIN_BATCH_SIZE_PER_GPU", defaults, int)
    config["inference_batch_size_per_gpu"] = get_val(['training', 'inference_batch_size_per_gpu'], "INFERENCE_BATCH_SIZE_PER_GPU", defaults, int)
    config["num_train_epochs"] = get_val(['training', 'num_epochs'], "NUM_TRAIN_EPOCHS", defaults, int)
    
    config["learning_rate_adamw"] = get_val(['training', 'optimizer', 'adamw', 'lr'], "LEARNING_RATE_ADAMW", defaults, float)
    adamw_betas_yaml = get_val(['training', 'optimizer', 'adamw', 'betas'], None, defaults) # Get list
    config["adamw_betas_0"] = float(os.getenv("ADAMW_BETAS_0", adamw_betas_yaml[0] if adamw_betas_yaml and len(adamw_betas_yaml)>0 else defaults.get('training',{}).get('optimizer',{}).get('adamw',{}).get('betas',[0.90,0.95])[0]))
    config["adamw_betas_1"] = float(os.getenv("ADAMW_BETAS_1", adamw_betas_yaml[1] if adamw_betas_yaml and len(adamw_betas_yaml)>1 else defaults.get('training',{}).get('optimizer',{}).get('adamw',{}).get('betas',[0.90,0.95])[1]))
    config["adamw_weight_decay"] = get_val(['training', 'optimizer', 'adamw', 'weight_decay'], "ADAMW_WEIGHT_DECAY", defaults, float)

    config["learning_rate_muon"] = get_val(['training', 'optimizer', 'muon', 'lr'], "LEARNING_RATE_MUON", defaults, float)
    config["muon_momentum"] = get_val(['training', 'optimizer', 'muon', 'momentum'], "MUON_MOMENTUM", defaults, float)
    
    # Add loading for the use_muon flag
    config["use_muon"] = get_val(['training', 'optimizer', 'use_muon'], "USE_MUON_OPTIMIZER", defaults, lambda x: str(x).lower() == 'true')    
    config["perform_training"] = get_val(['flags', 'perform_training'], "PERFORM_TRAINING", defaults, lambda x: str(x).lower() == 'true')
    config["perform_inference"] = get_val(['flags', 'perform_inference'], "PERFORM_INFERENCE", defaults, lambda x: str(x).lower() == 'true')
    
    os.makedirs(config["trained_model_output_dir"], exist_ok=True)
    os.makedirs(config["submission_dir"], exist_ok=True)
    return config