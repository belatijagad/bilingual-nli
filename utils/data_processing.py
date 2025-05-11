import pandas as pd
import torch
from torch.utils.data import TensorDataset

STRING_TO_INT_LABEL = {"c": 0, "n": 1, "e": 2}
INT_TO_STRING_LABEL = {v: k for k, v in STRING_TO_INT_LABEL.items()}
LABEL_NAMES_ORDERED = [INT_TO_STRING_LABEL[i] for i in sorted(INT_TO_STRING_LABEL.keys())]

def load_and_prepare_data(config, tokenizer):
    try:
        train_df_full = pd.read_csv(config["train_csv_path"])
        test_df_full = pd.read_csv(config["test_csv_path"])
    except FileNotFoundError as e:
        print(f"Error: Data file not found. {e}")
        print(f"Looked for TRAIN_CSV_PATH: {config['train_csv_path']}")
        print(f"Looked for TEST_CSV_PATH: {config['test_csv_path']}")
        raise

    test_df = test_df_full.loc[:,['id', 'final_statement_1', 'final_statement_2']].copy()
    test_df.columns = ['id', 'statement_1', 'statement_2']
    test_df["statement_1"] = test_df["statement_1"].astype(str)
    test_df["statement_2"] = test_df["statement_2"].astype(str)

    train_df = train_df_full.loc[:,['id', 'final_statement_1', 'final_statement_2', 'label']].copy()
    train_df.columns = ['id', 'statement_1', 'statement_2', 'label']
    train_df["statement_1"] = train_df["statement_1"].astype(str)
    train_df["statement_2"] = train_df["statement_2"].astype(str)

    train_dataset = None
    if config["perform_training"]:
        if 'label' not in train_df.columns:
            raise KeyError("Column 'label' not found in training data.")
            
        train_df['encoded_label'] = train_df['label'].map(STRING_TO_INT_LABEL)
        if train_df['encoded_label'].isnull().any():
            unknown = train_df[train_df['encoded_label'].isnull()]['label'].unique()
            raise ValueError(f"Unknown labels in train_df: {unknown}. Expected: {list(STRING_TO_INT_LABEL.keys())}")
        
        train_encodings = tokenizer(
            train_df['statement_1'].tolist(), train_df['statement_2'].tolist(),
            truncation=True, padding=True, max_length=config["max_length"], return_tensors="pt"
        )
        train_dataset = TensorDataset(
            train_encodings['input_ids'], train_encodings['attention_mask'],
            torch.tensor(train_df['encoded_label'].tolist())
        )
    return train_dataset, train_df, test_df