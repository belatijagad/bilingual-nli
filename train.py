import pandas as pd
import numpy as np
from datasets import Dataset, Features, Value, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import evaluate
import wandb
import huggingface_hub
import yaml
import os
from dotenv import load_dotenv
from tqdm import tqdm
from optimizer import setup_optimizers

def tokenize_function_for_map(examples, tokenizer, max_length, text_col1, text_col2):
    return tokenizer(
        examples[text_col1],
        examples[text_col2],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    predictions = np.argmax(logits, axis=-1)
    accuracy_metric = evaluate.load("accuracy")
    return accuracy_metric.compute(predictions=predictions, references=labels)

def evaluate_model(model, dataloader, loss_fn, device):
    model.eval()
    total_eval_loss = 0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            total_eval_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    avg_eval_loss = total_eval_loss / len(dataloader)
    
    # Concatenate all batches
    final_logits = torch.cat(all_logits)
    final_labels = torch.cat(all_labels)

    metrics = compute_metrics((final_logits, final_labels))
    return avg_eval_loss, metrics


def main(config_path="config.yaml"):
    load_dotenv()

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    model_config = config['model']
    tokenizer_config = config['tokenizer']
    training_config = config['training']
    hub_config = config['hub']
    wandb_config = config['wandb']

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # -- Dataset loading
    print(f"Loading data from: {data_config['train_path']}")
    full_df = pd.read_csv(data_config['train_path'])
    full_df = full_df[[data_config['raw_statement1_col'], data_config['raw_statement2_col'], data_config['raw_label_col']]]
    full_df.rename(columns={
        data_config['raw_statement1_col']: data_config['statement1_col'],
        data_config['raw_statement2_col']: data_config['statement2_col'],
        data_config['raw_label_col']: data_config['label_col']
    }, inplace=True)

    full_hf_dataset = Dataset.from_pandas(full_df)

    print(f"Casting features. Label column: '{data_config['label_col']}', with names: {model_config['label_names']}")
    features = Features({
        data_config['statement1_col']: Value("string"),
        data_config['statement2_col']: Value("string"),
        data_config['label_col']: ClassLabel(names=model_config['label_names'])
    })
    full_hf_dataset_featured = full_hf_dataset.cast(features)

    # -- Tokenization
    print(f"Loading tokenizer from: {model_config['checkpoint']}")
    tokenizer = AutoTokenizer.from_pretrained(model_config['checkpoint'])

    print("Tokenizing dataset...")
    tokenized_dataset = full_hf_dataset_featured.map(
        tokenize_function_for_map,
        batched=True,
        fn_kwargs={
            'tokenizer': tokenizer,
            'max_length': tokenizer_config['max_length'],
            'text_col1': data_config['statement1_col'],
            'text_col2': data_config['statement2_col']
        }
    )

    if data_config['label_col'] in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.rename_column(data_config['label_col'], "labels")

    columns_to_remove = [data_config['statement1_col'], data_config['statement2_col'], "__index_level_0__"]
    actual_columns_to_remove = [col for col in columns_to_remove if col in tokenized_dataset.column_names]
    
    if actual_columns_to_remove:
        tokenized_dataset = tokenized_dataset.remove_columns(actual_columns_to_remove)

    # -- Split dataset
    split_dataset = tokenized_dataset.train_test_split(test_size=training_config['validation_split_size'])
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    
    # -- Label config
    id2label = {i: name for i, name in enumerate(model_config['label_names'])}
    label2id = {name: i for i, name in enumerate(model_config['label_names'])}

    # -- Model loading
    print(f"Loading model from: {model_config['checkpoint']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config['checkpoint'],
        num_labels=model_config['num_labels'],
        id2label=id2label,
        label2id=label2id
    ).to(DEVICE)

    if training_config.get("use_torch_compile", False):
        print("Compiling the model with torch.compile()...")
        model = torch.compile(model)

    # -- Training preparation
    print("Setting up Optimizers...")
    optimizers_list = setup_optimizers(training_config, model, rank=0, world_size=1)
    loss_fn = nn.CrossEntropyLoss()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config['per_device_train_batch_size'],
        collate_fn=data_collator,
        shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_config['per_device_eval_batch_size'],
        collate_fn=data_collator
    )

    # -- Authenticate to hf and wandb
    hf_token = os.getenv("HF_TOKEN")
    if hf_token: huggingface_hub.login(token=hf_token)
    wandb_token = os.getenv("WANDB_TOKEN")
    if wandb_token: wandb.login(key=wandb_token)

    if 'wandb' in training_config['report_to']:
        print(f"Initializing WandB for project: {wandb_config['project']}")
        wandb.init(project=wandb_config['project'], reinit=wandb_config.get('reinit', True), config=config)

    # -- Training Loop & CSV Logging Setup
    training_history = []
    csv_log_path = training_config.get("csv_log_path")
    if csv_log_path:
        # Create directory for log file if it doesn't exist
        os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)
        print(f"Training logs will be saved to {csv_log_path}")

    print("Starting training...")
    for epoch in range(training_config['num_train_epochs']):
        model.train()
        total_train_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{training_config['num_train_epochs']}")
        for batch in progress_bar:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            for optimizer in optimizers_list:
                optimizer.zero_grad()
            
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            for optimizer in optimizers_list:
                optimizer.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            if 'wandb' in training_config['report_to']:
                wandb.log({"train_batch_loss": loss.item()})

        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # --- Evaluation Step ---
        avg_eval_loss, eval_metrics = evaluate_model(model, eval_dataloader, loss_fn, DEVICE)
        
        print(f"Epoch {epoch + 1} | Avg Train Loss: {avg_train_loss:.4f} | Avg Eval Loss: {avg_eval_loss:.4f} | Accuracy: {eval_metrics['accuracy']:.4f}")

        # --- Logging ---
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "eval_loss": avg_eval_loss,
            "accuracy": eval_metrics['accuracy']
        }
        
        if 'wandb' in training_config['report_to']:
            wandb.log(log_entry)
        
        # Append to history for CSV logging
        training_history.append(log_entry)
        
        # Save to CSV after each epoch
        if csv_log_path:
            pd.DataFrame(training_history).to_csv(csv_log_path, index=False)


    print("Training finished.")
    if 'wandb' in training_config['report_to']:
        wandb.finish()

    # --- Final Model Saving ---
    output_dir = training_config['output_dir']
    print(f"Saving model locally to: {output_dir}")
    # If using torch.compile, you need to save the original model if available
    model_to_save = getattr(model, '_orig_mod', model)
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model and tokenizer saved locally.")

    if hub_config.get('push_to_hub', False):
        print("Pushing model to hub...")
        model_to_save.push_to_hub(hub_config['model_id'])
        tokenizer.push_to_hub(hub_config['model_id'])

if __name__ == "__main__":
    main()