#!/usr/bin/env python3
"""
CRI_caculate.py - Calculate Cancer Risk Index (CRI) using trained TCRCan model.

This script reads TCR sequencing files (CSV format with 'cdr3aa' and 'fre' columns)
from a specified folder, uses a fine-tuned ESM-2 model to predict cancer-associated
TCRs, and computes CRI as the sum of frequencies of sequences with prediction
probability > 0.635. Results are appended to a global results CSV.
"""

import os
import gc
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from tqdm import tqdm

# ==================== Configuration (modify as needed) ====================
TOKENIZER_PATH = "../Rostlab/esm2_t30_150M_UR50D"          # Path to ESM-2 tokenizer
MODEL_PATH = "./esm2_t30_150M_UR50D13/1e-06_esm2_models_during_training/esm2_finetuning"  # Trained model
INPUT_FOLDER = "./data/test files/"                         # Folder containing input CSV files
OUTPUT_CSV = "../results.csv"                                # Global results file
PROBA_THRESHOLD = 0.635                                      # Threshold for cancer-associated classification
MAX_LENGTH = 512                                             # Max token length for ESM-2
BATCH_SIZE = 32                                              # Batch size for inference
# ==========================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def create_dataset(tokenizer, df, max_length=512):
    """
    Convert DataFrame with CDR3 sequences and frequencies into a tokenized Dataset.

    Args:
        tokenizer: ESM-2 tokenizer
        df: DataFrame with columns ['cdr3aa', 'fre']
        max_length: Maximum token length

    Returns:
        HuggingFace Dataset with input_ids, attention_mask, and original 'fre'
    """
    seqs_df = df.copy()
    # ESM tokenizer expects amino acids separated by spaces
    seqs_df['cdr3aa_input'] = seqs_df['cdr3aa'].apply(lambda x: " ".join(x))

    tokenized = tokenizer(
        list(seqs_df['cdr3aa_input']),
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    dataset = Dataset.from_dict(tokenized)
    dataset = dataset.add_column("cdr3aa", seqs_df['cdr3aa'])
    dataset = dataset.add_column("fre", seqs_df['fre'].astype(float))
    dataset = dataset.with_format("torch")
    return dataset

def main():
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
    model.to(device)
    model.eval()

    # Get list of input files
    all_files = [f.replace(".csv", "") for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv")]
    if not all_files:
        print(f"No CSV files found in {INPUT_FOLDER}")
        return
    print(f"Found {len(all_files)} files to process.")

    # Process each file
    for file in tqdm(all_files, desc="Processing files"):
        file_path = os.path.join(INPUT_FOLDER, file + ".csv")
        try:
            data0 = pd.read_csv(file_path, sep=',', encoding="GBK")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Rename columns to expected names (assumes first column is CDR3, second is frequency)
        data0.columns = ["cdr3aa", "fre"]

        # Filter sequences with frequency >= 3 * minimum frequency in this sample
        min_fre = data0["fre"].min()
        data = data0[data0["fre"] >= 3 * min_fre].copy()
        if data.empty:
            print(f"Warning: No sequences passed frequency filter in {file}")
            # Write empty result for this file
            result_row = pd.DataFrame({
                "file_name": [file],
                "1e-6_CRI_cutoff": [0.0]
            })
            # Append to global CSV
            if not os.path.exists(OUTPUT_CSV):
                result_row.to_csv(OUTPUT_CSV, mode='w', index=False, header=True)
            else:
                result_row.to_csv(OUTPUT_CSV, mode='a', index=False, header=False)
            continue

        # Create dataset and dataloader
        dataset = create_dataset(tokenizer, data, max_length=MAX_LENGTH)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        # Collect selected frequencies
        selected_fre = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                fre = batch['fre'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                proba = torch.softmax(logits, dim=1)
                # Get indices where probability for class 1 exceeds threshold
                mask = proba[:, 1] > PROBA_THRESHOLD
                selected_fre.extend(fre[mask].cpu().numpy())

        # Compute CRI as sum of frequencies of selected sequences
        cri = float(np.sum(selected_fre)) if selected_fre else 0.0

        # Prepare result row
        result_row = pd.DataFrame({
            "file_name": [file],
            "1e-6_CRI_cutoff": [cri]
        })

        # Append to global CSV (create if not exists)
        if not os.path.exists(OUTPUT_CSV):
            result_row.to_csv(OUTPUT_CSV, mode='w', index=False, header=True)
        else:
            result_row.to_csv(OUTPUT_CSV, mode='a', index=False, header=False)

    print(f"All results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()