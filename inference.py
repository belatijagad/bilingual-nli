import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import yaml
import os
from dotenv import load_dotenv
from tqdm import tqdm
import math

def load_app_config(config_path="config.yaml"):
  if os.path.exists(config_path):
    with open(config_path, 'r') as f:
      config = yaml.safe_load(f)
    return config
  print(f"Warning: Configuration file {config_path} not found. Using default values or command line arguments.")
  return {}

def main():
  load_dotenv()

  parser = argparse.ArgumentParser(description="Batch NLI Inference Script")
  parser.add_argument("--model_path_or_id", type=str, required=True, help="Path or Hugging Face Hub ID of the fine-tuned NLI model.")
  parser.add_argument("--input_csv_path", type=str, required=True, help="Path to the input CSV file containing test data.")
  parser.add_argument("--output_verbose_csv", type=str, default="predictions_verbose.csv", help="Filename for the verbose output CSV.")
  parser.add_argument("--output_submission_csv", type=str, default="submission.csv", help="Filename for the submission format CSV.")
  parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to the configuration YAML file.")
  
  args = parser.parse_args()
  
  app_config = load_app_config(args.config_path)

  data_cfg = app_config.get('data', {})
  tokenizer_cfg = app_config.get('tokenizer', {})
  inference_cfg = app_config.get('inference', {})

  input_id_col = data_cfg.get('test_input_id_col', 'id')
  input_statement1_col = data_cfg.get('test_input_statement1_col', 'statement_1')
  input_statement2_col = data_cfg.get('test_input_statement2_col', 'statement_2')
  
  max_length = tokenizer_cfg.get('max_length', 64)
  inference_batch_size = inference_cfg.get('batch_size', 32)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  num_gpus = torch.cuda.device_count()
  print(f"Using device: {device}, Number of GPUs: {num_gpus}")

  tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_id)
  model = AutoModelForSequenceClassification.from_pretrained(args.model_path_or_id)

  model.to(device)
  model.eval()

  test_df = pd.read_csv(args.input_csv_path)

  test_df['premise'] = test_df[input_statement1_col]
  test_df['hypothesis'] = test_df[input_statement2_col]

  results = []
  
  actual_model_config = model.module.config if isinstance(model, nn.DataParallel) else model.config
  id2label = actual_model_config.id2label
  
  label_names_ordered = [id2label[i] for i in sorted(id2label.keys())]

  print(f"Starting prediction with batch size {inference_batch_size} and max length {max_length}...")
  for i in tqdm(range(0, len(test_df), inference_batch_size), total=math.ceil(len(test_df)/inference_batch_size), desc="Predicting"):
    batch_slice = slice(i, min(i + inference_batch_size, len(test_df)))
    
    batch_ids = test_df[input_id_col].iloc[batch_slice].tolist()
    batch_premises = test_df["premise"].iloc[batch_slice].tolist()
    batch_hypotheses = test_df["hypothesis"].iloc[batch_slice].tolist()
    
    inputs = tokenizer(batch_premises, batch_hypotheses, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
      outputs = model(**inputs)
      logits = outputs.logits
      probs_tensor = torch.softmax(logits, dim=1)
      _ , pred_indices = torch.max(probs_tensor, dim=1)
        
    pred_indices_list = pred_indices.cpu().tolist()
    all_probs_list = probs_tensor.cpu().tolist()
    
    for batch_item_idx in range(len(batch_premises)):
      item_probabilities = all_probs_list[batch_item_idx]
      result_entry = {
        "id": batch_ids[batch_item_idx],
        "premise": batch_premises[batch_item_idx],
        "hypothesis": batch_hypotheses[batch_item_idx],
        "prediction": id2label[pred_indices_list[batch_item_idx]]
      }
      for class_idx, label_name_str in enumerate(label_names_ordered):
        result_entry[f"prob_{label_name_str}"] = item_probabilities[class_idx]
      results.append(result_entry)

  results_df = pd.DataFrame(results)
  
  prob_cols = [f"prob_{name}" for name in label_names_ordered]
  verbose_cols = ["id", "premise", "hypothesis", "prediction"] + prob_cols
  
  submission_verbose_df = results_df[verbose_cols]
  missing_cols_verbose = [col for col in verbose_cols if col not in results_df.columns]
  if missing_cols_verbose:
    raise KeyError(f"Columns {missing_cols_verbose} not found in results_df for verbose output. Available columns: {results_df.columns.tolist()}")
  
  submission_verbose_df.to_csv(args.output_verbose_csv, index=False)
  print(f"\nVerbose predictions saved to '{args.output_verbose_csv}'!")

  formatted_submission_df = pd.DataFrame({
    'id': results_df['id'],
    'label': results_df['prediction']
  })
  
  if input_id_col in test_df.columns and pd.api.types.is_integer_dtype(test_df[input_id_col].dtype):
    formatted_submission_df['id'] = formatted_submission_df['id'].astype(int)
      
  formatted_submission_df.to_csv(args.output_submission_csv, index=False)
  print(f"Formatted predictions saved to '{args.output_submission_csv}'!")

  print(f"\n'{args.output_verbose_csv}' head:\n{submission_verbose_df.head()}")
  print(f"\n'{args.output_submission_csv}' head:\n{formatted_submission_df.head()}")
  print("\n--- Script Finished ---")

if __name__ == "__main__":
  main()