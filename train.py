import pandas as pd
import numpy as np
from datasets import Dataset, Features, Value, ClassLabel
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding
)
import torch
import evaluate
import wandb
import huggingface_hub
import yaml
import os
from dotenv import load_dotenv

def tokenize_function_for_map(examples, tokenizer, max_length, text_col1, text_col2):
  return tokenizer(
    examples[text_col1],
    examples[text_col2],
    truncation=True,
    padding="max_length",
    max_length=max_length,
  )

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
  train_df = pd.read_csv(data_config['train_path'])
  train_df = train_df[[data_config['raw_statement1_col'], data_config['raw_statement2_col'], data_config['raw_label_col']]]
  train_df.rename(columns={
    data_config['raw_statement1_col']: data_config['statement1_col'],
    data_config['raw_statement2_col']: data_config['statement2_col'],
    data_config['raw_label_col']: data_config['label_col']
  }, inplace=True)

  train_hf_dataset = Dataset.from_pandas(train_df)

  print(f"Casting features. Label column: '{data_config['label_col']}', with names: {model_config['label_names']}")
  features = Features({
    data_config['statement1_col']: Value("string"),
    data_config['statement2_col']: Value("string"),
    data_config['label_col']: ClassLabel(names=model_config['label_names'])
  })
  train_hf_dataset_featured = train_hf_dataset.cast(features)

  # -- Tokenization
  print(f"Loading tokenizer from: {model_config['checkpoint']}")
  tokenizer = AutoTokenizer.from_pretrained(model_config['checkpoint'])

  print("Tokenizing dataset...")
  tokenized_train_dataset = train_hf_dataset_featured.map(
    tokenize_function_for_map,
    batched=True,
    fn_kwargs={
      'tokenizer': tokenizer,
      'max_length': tokenizer_config['max_length'],
      'text_col1': data_config['statement1_col'],
      'text_col2': data_config['statement2_col']
    }
  )

  if data_config['label_col'] in tokenized_train_dataset.column_names:
    tokenized_train_dataset = tokenized_train_dataset.rename_column(data_config['label_col'], "labels")
  elif "labels" not in tokenized_train_dataset.column_names:
    print(f"Warning: Column '{data_config['label_col']}' not found for renaming. Ensure 'labels' column exists if required by Trainer.")

  columns_to_remove = [data_config['statement1_col'], data_config['statement2_col']]
  actual_columns_to_remove = []
  for col in columns_to_remove:
    if col in tokenized_train_dataset.column_names and isinstance(tokenized_train_dataset[0][col], str):
      actual_columns_to_remove.append(col)

  if "__index_level_0__" in tokenized_train_dataset.column_names:
    actual_columns_to_remove.append("__index_level_0__")

  if actual_columns_to_remove:
    print(f"Removing columns: {actual_columns_to_remove}")
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(actual_columns_to_remove)
  
  tokenized_train_dataset.set_format("torch")
  train_dataset = tokenized_train_dataset

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

  # -- Training preparation
  print("Setting up Training Arguments...")
  training_args_dict = {
    "output_dir": training_config['output_dir'],
    "num_train_epochs": training_config['num_train_epochs'],
    "per_device_train_batch_size": training_config['per_device_train_batch_size'],
    "per_device_eval_batch_size": training_config['per_device_eval_batch_size'],
    "learning_rate": training_config['learning_rate'],
    "weight_decay": training_config['weight_decay'],
    "save_strategy": training_config['save_strategy'],
    "logging_dir": training_config['logging_dir'],
    "logging_steps": training_config['logging_steps'],
    "report_to": training_config['report_to'],
    "push_to_hub": hub_config['push_to_hub'],
    "hub_model_id": hub_config['model_id'],
    "hub_strategy": hub_config['strategy'],
  }
  if 'save_steps' in training_config:
    training_args_dict['save_steps'] = training_config['save_steps']
  if 'eval_strategy' in training_config:
    training_args_dict['evaluation_strategy'] = training_config['eval_strategy']
  if 'eval_steps' in training_config:
    training_args_dict['eval_steps'] = training_config['eval_steps']
  if 'metric_for_best_model' in training_config:
    training_args_dict['metric_for_best_model'] = training_config['metric_for_best_model']
  if 'load_best_model_at_end' in training_config:
    training_args_dict['load_best_model_at_end'] = training_config['load_best_model_at_end']

  training_arguments = TrainingArguments(**training_args_dict)

  metric = evaluate.load("accuracy")

  def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
  )
  
  # -- Authenticate to hf and wandb
  hf_token = os.getenv("HF_TOKEN"); huggingface_hub.login(token=hf_token)
  wandb_token = os.getenv("WANDB_TOKEN"); wandb.login(key=wandb_token)

  print(f"Initializing WandB for project: {wandb_config['project']}")
  wandb.init(project=wandb_config['project'], reinit=wandb_config.get('reinit', True), config=config)

  print("Starting training...")
  trainer.train()
  print("Training finished.")
  wandb.finish()

  print(f"Saving model locally to: {training_arguments.output_dir}")
  trainer.save_model(training_arguments.output_dir)
  tokenizer.save_pretrained(training_arguments.output_dir)
  print("Model and tokenizer saved locally.")

  trainer.push_to_hub()
  
if __name__ == "__main__":
  main()