# inference/predictor.py
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd

from utils.data_processing import INT_TO_STRING_LABEL, LABEL_NAMES_ORDERED 

def run_inference_engine(config, model_to_infer, tokenizer, test_df, device, is_main_process):
    if not is_main_process:
        return

    print("\n--- Starting Inference ---")
    model_to_infer.eval() 

    all_predictions_indices = []
    all_probabilities = []
    all_ids = test_df['id'].tolist()

    test_encodings = tokenizer(
        test_df['statement_1'].tolist(), 
        test_df['statement_2'].tolist(),
        truncation=True, 
        padding="max_length", 
        max_length=config["max_length"], 
        return_tensors="pt"
    )
    test_dataset_inf = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'])
    
    test_dataloader_inf = DataLoader(
        test_dataset_inf, 
        batch_size=config["inference_batch_size_per_gpu"], 
        shuffle=False # No need to shuffle for inference
    )

    with torch.no_grad():
        for batch in tqdm(test_dataloader_inf, desc="Inference", disable=not is_main_process):
            input_ids, attention_mask = [b.to(device, non_blocking=True) for b in batch]
            
            outputs = model_to_infer(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            probabilities_tensor = torch.softmax(logits, dim=-1)
            predictions_indices_tensor = torch.argmax(logits, dim=-1)
            
            all_predictions_indices.extend(predictions_indices_tensor.cpu().numpy())
            all_probabilities.extend(probabilities_tensor.cpu().numpy())

    predicted_labels_str = [INT_TO_STRING_LABEL[pred_idx] for pred_idx in all_predictions_indices]
    
    results_df_verbose = pd.DataFrame({
        'id': all_ids, 
        'prediction': predicted_labels_str
    })
    for i, label_name in enumerate(LABEL_NAMES_ORDERED):
        results_df_verbose[f'prob_{label_name}'] = [p[i] for p in all_probabilities]
    
    output_filename_verbose = os.path.join(config["submission_dir"], "submission_verbose.csv")
    results_df_verbose.to_csv(output_filename_verbose, index=False)
    print(f"Verbose submission file saved to {output_filename_verbose}")

    submission_df = results_df_verbose[['id', 'prediction']].copy()
    submission_df.columns = ['id', 'label']
    output_filename_submission = os.path.join(config["submission_dir"], "submission.csv")
    submission_df.to_csv(output_filename_submission, index=False)
    print(f"Submission file saved to {output_filename_submission}")
    print("--- Inference Finished ---")