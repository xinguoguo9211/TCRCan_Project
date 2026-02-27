"""
TCRCan: Fine-tuning ESM-2 for TCR Sequence Classification
This script trains an ESM-2 model on CDR3 sequences for binary classification
(cancer-associated vs non-cancer TCRs).
"""
import os
import gc
import shutil
import pickle
import pandas as pd
import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback
)
from evaluate import load

# Set environment variables for better compatibility
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)

def create_dataset(tokenizer, df, label_column='label', max_length=512):
    """
    Create tokenized dataset from CDR3 sequences

    Args:
        tokenizer: ESM-2 tokenizer
        df: DataFrame containing CDR3 sequences
        label_column: Name of the label column
        max_length: Maximum sequence length

    Returns:
        tokenized Dataset object
    """
    seqs_df = df.copy()
    # Add spaces between amino acids for tokenization (required by ESM tokenizer)
    seqs_df['cdr3aa_input'] = seqs_df['cdr3aa'].apply(lambda x: " ".join(x))

    # Tokenize sequences
    tokenized = tokenizer(
        list(seqs_df['cdr3aa_input']),
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    # Create HuggingFace Dataset
    dataset = Dataset.from_dict(tokenized)
    dataset = dataset.add_column("labels", list(seqs_df[label_column]))
    dataset = dataset.with_format("torch")

    return dataset

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics (accuracy and AUC)

    Args:
        eval_pred: Tuple of predictions and labels

    Returns:
        Dictionary of metrics
    """
    metric_acc = load('accuracy')
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = metric_acc.compute(predictions=preds, references=labels)['accuracy']

    # Calculate AUC (if binary classification)
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, predictions[:, 1])
    except:
        auc = 0.0

    return {"accuracy": acc, "auc": auc}

def main():
    """Main training function"""
    # Configuration
    DATA_PATH = "../data/Example.csv"          # Path to input CSV
    MODEL_NAME = "../Rostlab/esm2_t30_150M_UR50D"  # Path to pre-trained ESM-2 model
    OUTPUT_DIR = "../models/TCRCan"             # Base output directory
    LEARNING_RATE = 1e-6                         # Fixed learning rate
    MAX_LENGTH = 512                              # Max token length
    BATCH_SIZE = 32                               # Batch size per device
    EPOCHS = 5                                     # Number of training epochs
    SEED = 42

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_output_dir = os.path.join(OUTPUT_DIR, f"lr{LEARNING_RATE}_esm2_finetuned")
    os.makedirs(model_output_dir, exist_ok=True)

    # Set seeds
    set_seeds(SEED)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()

    # Load data
    print("Loading data...")
    data = pd.read_csv(DATA_PATH, sep=',', encoding="GBK")
    # Ensure required columns exist
    required_cols = ['cdr3aa', 'label', 'dataset']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data file.")

    # Split into train and validation based on 'dataset' column
    train_data = data[data['dataset'] == 'train'].reset_index(drop=True)
    valid_data = data[data['dataset'] == 'valid'].reset_index(drop=True)
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(valid_data)}")

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
    model.to(device)

    # Create datasets
    print("Tokenizing datasets...")
    train_dataset = create_dataset(tokenizer, train_data, 'label', MAX_LENGTH)
    val_dataset = create_dataset(tokenizer, valid_data, 'label', MAX_LENGTH)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        evaluation_strategy='epoch',          # Evaluate at each epoch
        logging_strategy='epoch',              # Log at each epoch
        save_strategy='epoch',
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        seed=SEED,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        gradient_accumulation_steps=2,         # Simulate larger batch
        save_total_limit=2,                     # Keep only best and last
        remove_unused_columns=False,            # Keep all columns in dataset
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        # Uncomment for early stopping
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model and tokenizer
    final_model_path = os.path.join(model_output_dir, 'best_model')
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Best model saved to {final_model_path}")

    # Save training history
    history_path = os.path.join(model_output_dir, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(trainer.state.log_history, f)

    # Clean up checkpoint directories to save space
    for item in os.listdir(model_output_dir):
        item_path = os.path.join(model_output_dir, item)
        if os.path.isdir(item_path) and "checkpoint" in item:
            try:
                shutil.rmtree(item_path)
                print(f"Deleted checkpoint: {item_path}")
            except Exception as e:
                print(f"Failed to delete {item_path}: {e}")

    print("Training completed successfully!")

if __name__ == "__main__":
    main()