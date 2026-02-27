# TCRCan: ESM-2 based Deep Learning Framework for Pan-cancer Prediction

## Overview
TCRCan is a deep learning framework that fine-tunes the ESM-2 protein language model to classify T-cell receptor (TCR) sequences as cancer-associated or non-cancer. The model then calculates a Cancer Risk Index (CRI) for individual samples by aggregating the frequencies of predicted cancer-associated TCRs. TCRCan supports pan-cancer detection using peripheral blood TCRβ repertoires.

## Key Features
- Fine-tunes ESM-2 (esm2_t30_150M_UR50D) on CDR3 sequences of any length (padded/truncated to 512 tokens)
- Single model for all sequences, no length-specific training
- Computes CRI based on predicted probabilities and observed TCR frequencies
- Includes separate scripts for training and sample-level prediction
- Automatically removes intermediate checkpoints to save disk space

## Project Structure
TCRCan_Project/

├── data/

│ ├── Example.csv # Training/validation data with 'train'/'valid' split

│ └── test_files/ # Individual sample CSV files for prediction

├── models/ # Output directory for trained models (created automatically)

├── scripts/

│ ├── TCRCan.py # Main training script

│ └── CRI_caculate.py # CRI calculation script for test samples

├── requirements.txt # Python dependencies

└── README.md # This file


## Development Environment
- Python 3.10.14
- CUDA Version: 11.4
- transformers: 4.24.0
- torch: 1.12.1

## 1. Data Preparation

### Training/Validation Data (`Example.csv`)
CSV file with the following columns:
- `cdr3aa`: CDR3 amino acid sequence (string, e.g., `CASSLGQGYEQYF`)
- `label`: Binary label (0 = non-cancer, 1 = cancer-associated)
- `dataset`: Data split indicator, must be either `"train"` or `"valid"`

Example format:

cdr3aa,label,dataset
CASSLGQGYEQYF,1,train

CASSQDRLGKNIQYF,0,valid

CASSYSTDTQYF,1,train

Each file corresponds to one patient/sample and must be a CSV with no header, containing two columns:

CDR3 amino acid sequence

Frequency (numeric, e.g., read count or abundance)

CASSLGQGYEQYF,15

CASSQDRLGKNIQYF,3

CASSYSTDTQYF,7

## 2. Model Training
Run the training script:
cd scripts
python TCRCan.py
## 3. Prediction (CRI Calculation)
After training, use CRI_caculate.py to compute CRI scores for all test sample files placed in ../data/test_files/.
cd scripts
python CRI_caculate.py
The prediction script generates a single CSV file (results.csv) with two columns:

file_name: name of the test file (without extension)

1e-6_CRI_cutoff: calculated CRI value

Example results.csv:
file_name,1e-6_CRI_cutoff
patient1,25.0
patient2,3.5
patient3,0.0

## Pre-trained Model
TCRCan uses the ESM-2 model facebook/esm2_t30_150M_UR50D as the base. The tokenizer and model are loaded from a local path (../Rostlab/esm2_t30_150M_UR50D in the scripts). Adjust the path if you store the model elsewhere or prefer to download directly from Hugging Face Hub (replace with "facebook/esm2_t30_150M_UR50D").
