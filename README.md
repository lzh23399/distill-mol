# Distill-Mol

Distill-Mol is a multimodal molecular representation learning framework for molecular property prediction and drug-target pair prediction. The model integrates molecular information from three complementary views:

- **1D semantic view**: SMILES sequence representation from a pretrained molecular language model.
- **2D topological view**: molecular graph representation learned by a GAT encoder.
- **3D geometric view**: conformer representation learned with teacher-student geometry distillation.

The three modality-specific representations are fused by a dynamic cross-modal fusion module for downstream prediction.

## Repository Scope

This repository contains **code only**. The following files are intentionally not included:

- MoleculeNet, ZINC, DTA, or DTI datasets.
- Pretrained language model folders, such as `PubChem10M_SMILES_BPE_450k/`.
- ESM-2 folders, such as `esm2_t30_150M_UR50D/`.
- Pretraining checkpoints and downstream fine-tuned checkpoints.
- Cached preprocessing files and experimental outputs.

## Directory Structure

```text
distill/
|-- data/
|   |-- data_process.py          # MoleculeNet preprocessing
|   `-- zinc_process.py          # ZINC preprocessing for pretraining
|-- pretrain/
|   |-- pretrain_2D.py           # 2D graph masked pretraining
|   |-- pretrain_egnn_zinc.py    # 3D teacher-student geometry distillation
|   `-- models/
|       |-- teacher_model.py
|       `-- student_model.py
|-- egnn-pytorch-main/           # Local EGNN dependency source
|-- fusion_model.py              # 1D/2D/3D encoders and cross-modal fusion
|-- down.py                      # MoleculeNet downstream training
|-- distill_pair_down.py         # DTA/DTI downstream training
`-- requirements.txt
```

## Installation

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

The provided `requirements.txt` is aligned with the tested environment:

```text
Python 3.12
PyTorch 2.2.0
CUDA 12.1
PyTorch Geometric 2.7.0
```

If your CUDA or PyTorch version differs, install the matching PyTorch Geometric wheels from the official PyG website.

## Required External Resources

Before running downstream tasks, prepare the external model folders manually:

```text
PubChem10M_SMILES_BPE_450k/      # Molecular language model
esm2_t30_150M_UR50D/             # ESM-2 protein model for DTA/DTI
```

Download the ESM-2 model separately from HuggingFace or provide a local ESM-2 directory with `--protein_model_path`.

## ZINC Preprocessing for Pretraining

The pretraining scripts expect preprocessed ZINC files, including `2d_graphs.pkl`, `3d_conformers.pkl`, and `metadata.pkl`. Use `data/zinc_process.py` to convert a ZINC CSV file into this format.

The input CSV should contain a SMILES column. The script automatically checks common SMILES column names.

```bash
python data/zinc_process.py \
  --zinc_csv ./data/zinc15/zinc15.csv \
  --output_dir ./data/zinc15/output \
  --num_workers 8 \
  --timeout 1.0
```

For a quick preprocessing test, limit the number of molecules:

```bash
python data/zinc_process.py \
  --zinc_csv ./data/zinc15/zinc15.csv \
  --output_dir ./data/zinc15/output/test \
  --max_samples 1000 \
  --num_workers 8 \
  --timeout 1.0
```

The output directory will contain:

```text
2d_graphs.pkl
3d_conformers.pkl
metadata.pkl
statistics.json
processing_report.txt
```

## Pretraining

### 2D Graph Pretraining

```bash
python pretrain/pretrain_2D.py \
  --data_dir ./data/zinc15/output \
  --checkpoint_dir ./pretrain/check_all \
  --epochs 30
```

### 3D Geometry Distillation

Train both the teacher and student models:

```bash
python pretrain/pretrain_egnn_zinc.py \
  --train_both \
  --teacher_epochs 50 \
  --student_epochs 100 \
  --output_dim 256 \
  --data_dir ./data/zinc15/output
```

The current code pretrains the 2D and 3D branches separately. The 1D branch is initialized from an external pretrained molecular language model.

## MoleculeNet Preprocessing

Example for a classification dataset:

```bash
python data/data_process.py \
  --csv_path ./data/molecular/cla/bbbp/bbbp.csv \
  --smiles_col smiles \
  --target_col p_np \
  --task_type classification \
  --output_dir ./data/molecular/cla/bbbp/process \
  --num_workers 8 \
  --timeout 5
```

For multi-task datasets, use:

```bash
--target_col all
```

To disable 3D conformer generation:

```bash
--no_3d
```

## MoleculeNet Downstream Training

```bash
python down.py \
  --preprocessed_dir ./data/molecular/cla/bbbp/process \
  --task_type classification \
  --split_type scaffold \
  --output_dir ./output/molecular/bbbp \
  --roberta_path ./PubChem10M_SMILES_BPE_450k
```

For regression tasks:

```bash
python down.py \
  --preprocessed_dir ./data/molecular/reg/esol/process \
  --task_type regression \
  --split_type scaffold \
  --output_dir ./output/molecular/esol \
  --roberta_path ./PubChem10M_SMILES_BPE_450k
```

## DTA Training

```bash
python distill_pair_down.py \
  --task dta \
  --csv_path ./data/Protein_Ligand/DTA/davis.csv \
  --protein_model_path ./esm2_t30_150M_UR50D \
  --roberta_path ./PubChem10M_SMILES_BPE_450k \
  --output_dir ./output/pair/davis \
  --batch_size 2 \
  --epochs 30
```

DTA metrics include MSE, RMSE, MAE, CI, R2, Pearson, and Spearman.

## DTI Training

```bash
python distill_pair_down.py \
  --task dti \
  --csv_path ./data/Protein_Ligand/DTI/celegans.csv \
  --protein_model_path ./esm2_t30_150M_UR50D \
  --roberta_path ./PubChem10M_SMILES_BPE_450k \
  --output_dir ./output/pair/dti_celegans \
  --batch_size 2 \
  --epochs 40
```

DTI metrics include accuracy, precision, recall, F1, and ROC-AUC.

## Notes

- Use `--no_3d` first if GPU memory is limited, especially for DTA/DTI.
- Do not use `--batch_size 1` with the current 2D GAT branch because BatchNorm requires more than one training sample per batch.
- The current pair-task split in `distill_pair_down.py` is warm-start random pair splitting unless additional cold-start split logic is added.
