from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data_processing import STRING_TO_INT_LABEL

def get_tokenizer(config):
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], token=config.get("hf_token"))
    except Exception as e:
        print(f"Tokenizer load error for '{config['model_name']}' (token: {bool(config.get('hf_token'))}): {e}")
        if config.get("hf_token"):
            print("Retrying tokenizer load without token...")
            tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        else: raise e
    return tokenizer

def get_model(config):
    num_labels = len(STRING_TO_INT_LABEL)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"], num_labels=num_labels, token=config.get("hf_token"))
    except Exception as e:
        print(f"Model load error for '{config['model_name']}' (token: {bool(config.get('hf_token'))}): {e}")
        if config.get("hf_token"):
            print("Retrying model load without token...")
            model = AutoModelForSequenceClassification.from_pretrained(config["model_name"], num_labels=num_labels)
        else: raise e
    return model

def wrap_model_ddp(model, device, world_size):
    if world_size > 1:
        # Ensure model is on the correct device BEFORE DDP wrapping
        model.to(device) 
        model = DDP(model, device_ids=[device.index if device.type == "cuda" else None], find_unused_parameters=False)
    else:
        model.to(device) # Still ensure model is on device for single process
    return model