"""Downstream training entry for multimodal molecular property prediction.

Examples:
    python down.py --preprocessed_dir ./data/molecular/cla/bbbp/process --task_type classification --split_type scaffold
    python down.py --preprocessed_dir ./data/molecular/reg/esol/process --task_type regression --split_type scaffold
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import argparse
import json
import pickle
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print('Warning: RDKit is not installed, scaffold split is unavailable.')
sys.path.append('.')
from fusion_model import MultiModalMoleculeModel, MultiModalPropertyPredictor

def get_scaffold(smiles):
    """Return the Bemis-Murcko scaffold SMILES for a molecule."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ''
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        return scaffold_smiles
    except Exception:
        return ''

def generate_scaffold_groups(smiles_list):
    """Group molecule indices by their Bemis-Murcko scaffolds."""
    scaffold_to_indices = defaultdict(list)
    for idx, smiles in enumerate(tqdm(smiles_list, desc='Generating scaffolds')):
        scaffold = get_scaffold(str(smiles))
        scaffold_to_indices[scaffold].append(idx)
    return scaffold_to_indices

def scaffold_split(smiles_list, labels_list=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, balanced=False):
    """Split molecules by scaffold groups into train, validation, and test sets."""
    if not RDKIT_AVAILABLE:
        raise ImportError('RDKit is required for scaffold split. Please install rdkit.')
    np.random.seed(seed)
    scaffold_to_indices = generate_scaffold_groups(smiles_list)
    scaffolds = sorted(scaffold_to_indices.keys(), key=lambda s: (len(scaffold_to_indices[s]), scaffold_to_indices[s][0]), reverse=True)
    n_total = len(smiles_list)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    train_idx, val_idx, test_idx = ([], [], [])
    for scaffold in scaffolds:
        indices = scaffold_to_indices[scaffold]
        if len(train_idx) + len(indices) <= n_train:
            train_idx.extend(indices)
        elif len(val_idx) + len(indices) <= n_val:
            val_idx.extend(indices)
        else:
            test_idx.extend(indices)
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    print('\nScaffold split statistics:')
    print(f'  Number of scaffolds: {len(scaffold_to_indices)}')
    print(f'  Train: {len(train_idx)} samples')
    print(f'  Val: {len(val_idx)} samples')
    print(f'  Test: {len(test_idx)} samples')
    return (np.array(train_idx), np.array(val_idx), np.array(test_idx))

def random_split(indices, labels=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, stratify=False, task_type='classification'):
    """Split sample indices randomly, optionally using stratified sampling."""
    n_total = len(indices)
    actual_test_ratio = test_ratio / (train_ratio + val_ratio + test_ratio)
    actual_val_ratio = val_ratio / (train_ratio + val_ratio)
    use_stratify = stratify and labels is not None and (task_type == 'classification') and (len(np.array(labels).shape) == 1)
    if use_stratify:
        labels_array = np.array(labels)
        stratify_labels = labels_array[indices] if len(labels_array) > len(indices) else labels_array
        train_val_idx, test_idx = train_test_split(indices, test_size=actual_test_ratio, random_state=seed, stratify=stratify_labels)
        train_val_labels = labels_array[train_val_idx]
        train_idx, val_idx = train_test_split(train_val_idx, test_size=actual_val_ratio, random_state=seed, stratify=train_val_labels)
    else:
        train_val_idx, test_idx = train_test_split(indices, test_size=actual_test_ratio, random_state=seed)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=actual_val_ratio, random_state=seed)
    print('\nRandom split statistics:')
    print(f'  Stratified: {use_stratify}')
    print(f'  Train: {len(train_idx)} samples')
    print(f'  Val: {len(val_idx)} samples')
    print(f'  Test: {len(test_idx)} samples')
    return (np.array(train_idx), np.array(val_idx), np.array(test_idx))

def split_dataset(smiles_list, labels_list, split_type='random', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, task_type='classification', stratify=True, balanced_scaffold=False):
    """Dispatch dataset splitting according to the requested split strategy."""
    n_samples = len(smiles_list)
    indices = np.arange(n_samples)
    print(f"\n{'=' * 60}")
    print(f'Split strategy: {split_type.upper()}')
    print(f"{'=' * 60}")
    print(f'Total samples: {n_samples}')
    print(f'Ratios: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}')
    if split_type == 'scaffold':
        train_idx, val_idx, test_idx = scaffold_split(smiles_list, labels_list, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed, balanced=balanced_scaffold)
    elif split_type == 'random':
        train_idx, val_idx, test_idx = random_split(indices, labels_list, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed, stratify=stratify, task_type=task_type)
    else:
        raise ValueError(f'Unsupported split_type: {split_type}; use random or scaffold')
    all_idx = set(train_idx) | set(val_idx) | set(test_idx)
    assert len(all_idx) == n_samples, 'Split sample count mismatch'
    assert len(set(train_idx) & set(val_idx)) == 0, 'Train and validation overlap'
    assert len(set(train_idx) & set(test_idx)) == 0, 'Train and test overlap'
    assert len(set(val_idx) & set(test_idx)) == 0, 'Validation and test overlap'
    return (train_idx, val_idx, test_idx)

class PreprocessedMultiModalDataset(Dataset):
    """PreprocessedMultiModalDataset implementation."""

    def __init__(self, indices, smiles_list, labels_list, graph_2d_list, conformer_3d_list, tokenizer, task_type, num_tasks, max_length=128, use_1d=True, use_2d=True, use_3d=True):
        """__init__ helper."""
        self.indices = indices
        self.smiles_list = smiles_list
        self.labels_list = labels_list
        self.graph_2d_list = graph_2d_list
        self.conformer_3d_list = conformer_3d_list
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.num_tasks = num_tasks
        self.max_length = max_length
        self.use_1d = use_1d
        self.use_2d = use_2d
        self.use_3d = use_3d
        print(f'Dataset size: {len(indices)} samples; modalities: 1D={use_1d}, 2D={use_2d}, 3D={use_3d}')
        print(f'Task type: {task_type}; number of tasks: {num_tasks}')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        smiles = str(self.smiles_list[real_idx])
        label = self.labels_list[real_idx]
        if self.task_type == 'classification':
            if self.num_tasks == 1:
                label_tensor = torch.tensor(label, dtype=torch.long)
            else:
                label_tensor = torch.tensor(label, dtype=torch.float)
        elif self.num_tasks == 1:
            label_tensor = torch.tensor(label, dtype=torch.float)
        else:
            label_tensor = torch.tensor(label, dtype=torch.float)
        sample = {'label': label_tensor, 'smiles': smiles, 'idx': real_idx}
        if self.use_1d:
            enc = self.tokenizer(smiles, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
            sample['1d'] = {'input_ids': enc['input_ids'].flatten(), 'attention_mask': enc['attention_mask'].flatten()}
        if self.use_2d:
            graph_2d = self.graph_2d_list[real_idx]
            sample['2d'] = Data(x=graph_2d['x'], edge_index=graph_2d['edge_index'], edge_attr=graph_2d['edge_attr'])
        if self.use_3d:
            conf_3d = self.conformer_3d_list[real_idx]
            sample['3d'] = {'atom_types': conf_3d['atom_types'], 'positions': conf_3d['pos'], 'num_atoms': conf_3d['num_nodes']}
        return sample

def collate_fn(batch):
    """Collate multimodal molecular samples into a mini-batch."""
    result = {'label': torch.stack([b['label'] for b in batch]), 'smiles': [b['smiles'] for b in batch], 'idx': [b['idx'] for b in batch]}
    if '1d' in batch[0]:
        result['1d'] = {'input_ids': torch.stack([b['1d']['input_ids'] for b in batch]), 'attention_mask': torch.stack([b['1d']['attention_mask'] for b in batch])}
    if '2d' in batch[0]:
        result['2d'] = Batch.from_data_list([b['2d'] for b in batch])
    if '3d' in batch[0]:
        max_n = max((b['3d']['positions'].shape[0] for b in batch))
        B = len(batch)
        atom_types = torch.zeros(B, max_n, dtype=torch.long)
        positions = torch.zeros(B, max_n, 3)
        mask = torch.zeros(B, max_n)
        for i, b in enumerate(batch):
            n = b['3d']['positions'].shape[0]
            atom_types[i, :n] = b['3d']['atom_types'][:n]
            positions[i, :n] = b['3d']['positions']
            mask[i, :n] = 1.0
        result['3d'] = {'atom_types': atom_types, 'positions': positions, 'mask': mask}
    return result

def train_epoch(model, loader, optimizer, scheduler, device, task_type, num_tasks):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    preds_list, probs_list, labels_list, weights_list = ([], [], [], [])
    if task_type == 'classification':
        if num_tasks == 1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss(reduction='none')
    else:
        criterion = nn.MSELoss() if num_tasks == 1 else nn.MSELoss(reduction='none')
    for batch in tqdm(loader, desc='Train'):
        label = batch['label'].to(device)
        batch_data = {}
        if '1d' in batch:
            batch_data['1d'] = {k: v.to(device) for k, v in batch['1d'].items()}
        if '2d' in batch:
            batch_data['2d'] = batch['2d'].to(device)
        if '3d' in batch:
            batch_data['3d'] = {k: v.to(device) for k, v in batch['3d'].items()}
        logits, weights = model(batch_data, return_weights=True)
        if task_type == 'classification':
            if num_tasks == 1:
                loss = criterion(logits, label)
            else:
                mask = ~torch.isnan(label) & (label >= 0) & (label <= 1)
                safe_label = torch.where(mask, label, torch.zeros_like(label))
                loss = criterion(logits, safe_label)
                loss = (loss * mask.float()).sum() / mask.sum().clamp_min(1)
        elif num_tasks == 1:
            loss = criterion(logits.squeeze(), label.squeeze())
        else:
            mask = ~torch.isnan(label)
            loss = criterion(logits, label)
            loss = (loss * mask.float()).sum() / mask.sum()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        if task_type == 'classification':
            if num_tasks == 1:
                preds_list.extend(logits.detach().argmax(1).cpu().numpy())
                probs_list.extend(F.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy())
            else:
                probs_list.append(torch.sigmoid(logits.detach()).cpu().numpy())
        else:
            preds_list.append(logits.detach().cpu().numpy())
        labels_list.append(label.cpu().numpy())
        weights_list.append(weights.cpu())
    avg_loss = total_loss / len(loader)
    avg_weights = torch.cat(weights_list).mean(0).detach().numpy()
    labels_arr = np.concatenate(labels_list, axis=0)
    if task_type == 'classification':
        if num_tasks == 1:
            acc = accuracy_score(labels_arr.flatten(), preds_list)
            return (avg_loss, {'accuracy': acc}, avg_weights)
        else:
            probs_arr = np.vstack(probs_list)
            roc_aucs = []
            for i in range(num_tasks):
                valid_mask = ~np.isnan(labels_arr[:, i]) & (labels_arr[:, i] >= 0) & (labels_arr[:, i] <= 1)
                if valid_mask.sum() > 0 and len(np.unique(labels_arr[valid_mask, i])) > 1:
                    try:
                        auc = roc_auc_score(labels_arr[valid_mask, i], probs_arr[valid_mask, i])
                        roc_aucs.append(auc)
                    except:
                        pass
            return (avg_loss, {'roc_auc': np.mean(roc_aucs) if roc_aucs else 0.5}, avg_weights)
    else:
        preds_arr = np.concatenate(preds_list, axis=0)
        if num_tasks == 1:
            rmse = np.sqrt(mean_squared_error(labels_arr.flatten(), preds_arr.flatten()))
        else:
            rmse = np.sqrt(np.nanmean((labels_arr - preds_arr) ** 2))
        return (avg_loss, {'rmse': rmse}, avg_weights)

@torch.no_grad()
def evaluate(model, loader, device, task_type, num_tasks):
    model.eval()
    if len(loader) == 0:
        print('Warning: empty data loader; check split ratios and dataset size.')
        if task_type == 'classification':
            return ({'accuracy': 0.0, 'roc_auc': 0.0}, np.zeros(3))
        else:
            return ({'rmse': 0.0, 'mae': 0.0}, np.zeros(3))
    preds_list, probs_list, labels_list, weights_list = ([], [], [], [])
    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval'):
            label = batch['label'].to(device)
            batch_data = {}
            if '1d' in batch:
                batch_data['1d'] = {k: v.to(device) for k, v in batch['1d'].items()}
            if '2d' in batch:
                batch_data['2d'] = batch['2d'].to(device)
            if '3d' in batch:
                batch_data['3d'] = {k: v.to(device) for k, v in batch['3d'].items()}
            logits, weights = model(batch_data, return_weights=True)
            if task_type == 'classification':
                if num_tasks == 1:
                    preds_list.extend(logits.argmax(1).cpu().numpy())
                    probs_list.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
                else:
                    probs_list.append(torch.sigmoid(logits).cpu().numpy())
            else:
                preds_list.append(logits.cpu().numpy())
            labels_list.append(label.cpu().numpy())
            weights_list.append(weights.cpu())
    if len(labels_list) == 0:
        print('Warning: no evaluation data were collected.')
        if task_type == 'classification':
            return ({'accuracy': 0.0, 'roc_auc': 0.0}, np.zeros(3))
        else:
            return ({'rmse': 0.0, 'mae': 0.0}, np.zeros(3))
    labels_arr = np.concatenate(labels_list, axis=0)
    avg_weights = torch.cat(weights_list).mean(0).numpy()
    metrics = {}
    if task_type == 'classification':
        if num_tasks == 1:
            labels_flat = labels_arr.flatten()
            metrics['accuracy'] = accuracy_score(labels_flat, preds_list)
            metrics['precision'] = precision_score(labels_flat, preds_list, zero_division=0)
            metrics['recall'] = recall_score(labels_flat, preds_list, zero_division=0)
            metrics['f1'] = f1_score(labels_flat, preds_list, zero_division=0)
            try:
                metrics['roc_auc'] = roc_auc_score(labels_flat, probs_list)
            except:
                metrics['roc_auc'] = 0.5
        else:
            probs_arr = np.vstack(probs_list)
            roc_aucs = []
            for i in range(num_tasks):
                valid_mask = ~np.isnan(labels_arr[:, i]) & (labels_arr[:, i] >= 0) & (labels_arr[:, i] <= 1)
                if valid_mask.sum() > 0 and len(np.unique(labels_arr[valid_mask, i])) > 1:
                    try:
                        auc = roc_auc_score(labels_arr[valid_mask, i], probs_arr[valid_mask, i])
                        roc_aucs.append(auc)
                    except:
                        pass
            metrics['roc_auc'] = np.mean(roc_aucs) if roc_aucs else 0.5
            metrics['num_valid_tasks'] = len(roc_aucs)
    else:
        preds_arr = np.concatenate(preds_list, axis=0)
        if num_tasks == 1:
            labels_flat = labels_arr.flatten()
            preds_flat = preds_arr.flatten()
            metrics['rmse'] = np.sqrt(mean_squared_error(labels_flat, preds_flat))
            metrics['mae'] = mean_absolute_error(labels_flat, preds_flat)
            metrics['r2'] = r2_score(labels_flat, preds_flat)
            if len(labels_flat) > 2:
                try:
                    metrics['pearson'], _ = pearsonr(labels_flat, preds_flat)
                    metrics['spearman'], _ = spearmanr(labels_flat, preds_flat)
                except:
                    pass
        else:
            rmses, maes = ([], [])
            for i in range(num_tasks):
                valid_mask = ~np.isnan(labels_arr[:, i])
                if valid_mask.sum() > 0:
                    rmses.append(np.sqrt(mean_squared_error(labels_arr[valid_mask, i], preds_arr[valid_mask, i])))
                    maes.append(mean_absolute_error(labels_arr[valid_mask, i], preds_arr[valid_mask, i]))
            metrics['rmse'] = np.mean(rmses) if rmses else 0.0
            metrics['mae'] = np.mean(maes) if maes else 0.0
    return (metrics, avg_weights)

def plot_curves(history, task_type, save_path):
    """plot_curves helper."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history['train_loss'])
    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Epoch')
    if task_type == 'classification':
        if 'train_acc' in history:
            axes[1].plot(history['train_acc'], label='Train')
            axes[1].plot(history['val_acc'], label='Val')
            axes[1].set_title('Accuracy')
        else:
            axes[1].plot(history.get('train_auc', []), label='Train')
            axes[1].plot(history.get('val_auc', []), label='Val')
            axes[1].set_title('ROC-AUC')
        axes[1].legend()
        axes[2].plot(history['val_metric'])
        axes[2].set_title('Val ROC-AUC')
    else:
        axes[1].plot(history['train_rmse'], label='Train')
        axes[1].plot(history['val_rmse'], label='Val')
        axes[1].set_title('RMSE')
        axes[1].legend()
        axes[2].plot(history['val_metric'])
        axes[2].set_title('Val RMSE')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_weights(weights_history, names, save_path):
    """Plot modality weight curves."""
    arr = np.array(weights_history)
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(names):
        plt.plot(arr[:, i], label=name, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    plt.title('Modal Weights')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    os.makedirs(args.output_dir, exist_ok=True)
    print('\n' + '=' * 60)
    print('Loading preprocessed multimodal data')
    print('=' * 60)
    print(f'\nData directory: {args.preprocessed_dir}')
    with open(os.path.join(args.preprocessed_dir, 'smiles.pkl'), 'rb') as f:
        smiles_list = pickle.load(f)
    print(f'SMILES: {len(smiles_list)}')
    with open(os.path.join(args.preprocessed_dir, 'labels.pkl'), 'rb') as f:
        labels_list = pickle.load(f)
    print(f'Labels: {len(labels_list)}')
    with open(os.path.join(args.preprocessed_dir, '2d_graphs.pkl'), 'rb') as f:
        graph_2d_list = pickle.load(f)
    print(f'2D graphs: {len(graph_2d_list)}')
    with open(os.path.join(args.preprocessed_dir, '3d_conformers.pkl'), 'rb') as f:
        conformer_3d_list = pickle.load(f)
    print(f'3D: {len(conformer_3d_list)}')
    config_path = os.path.join(args.preprocessed_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data_config = json.load(f)
        print('\nData configuration:')
        print(f"  Task type: {data_config.get('task_type', args.task_type)}")
        print(f"  Target columns: {data_config.get('target_cols', 'unknown')}")
        print(f"  Number of tasks: {data_config.get('num_tasks', 1)}")
        if args.task_type is None:
            args.task_type = data_config.get('task_type', 'classification')
        num_tasks = data_config.get('num_tasks', 1)
    else:
        print('Warning: config.json was not found; falling back to command-line arguments.')
        if args.task_type is None:
            args.task_type = 'classification'
        if isinstance(labels_list[0], (list, np.ndarray)):
            num_tasks = len(labels_list[0])
        else:
            num_tasks = 1
    print('\nTraining configuration:')
    print(f'  Task type: {args.task_type}')
    print(f'  Number of tasks: {num_tasks}')
    print(f'  Split type: {args.split_type}')
    if args.task_type == 'classification' and num_tasks == 1:
        pos_count = sum(labels_list)
        neg_count = len(labels_list) - pos_count
        print(f'\nClass distribution: negative={neg_count}, positive={pos_count}')
    elif args.task_type == 'regression' and num_tasks == 1:
        labels_arr = np.array(labels_list)
        print(f'\nLabel statistics: mean={labels_arr.mean():.4f}, std={labels_arr.std():.4f}')
    train_idx, val_idx, test_idx = split_dataset(smiles_list=smiles_list, labels_list=labels_list, split_type=args.split_type, train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed, task_type=args.task_type, stratify=args.stratify, balanced_scaffold=args.balanced_scaffold)
    print(f'\nTokenizer: {args.roberta_path}')
    try:
        tokenizer = RobertaTokenizer.from_pretrained(args.roberta_path, local_files_only=True)
        print('Loaded tokenizer from local path.')
    except Exception as e:
        print(f'Failed to load local tokenizer: {e}')
        tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        print('Loaded fallback ChemBERTa tokenizer.')
    common_kwargs = {'smiles_list': smiles_list, 'labels_list': labels_list, 'graph_2d_list': graph_2d_list, 'conformer_3d_list': conformer_3d_list, 'tokenizer': tokenizer, 'task_type': args.task_type, 'num_tasks': num_tasks, 'use_1d': args.use_1d, 'use_2d': args.use_2d, 'use_3d': args.use_3d}
    train_ds = PreprocessedMultiModalDataset(train_idx, **common_kwargs)
    val_ds = PreprocessedMultiModalDataset(val_idx, **common_kwargs)
    test_ds = PreprocessedMultiModalDataset(test_idx, **common_kwargs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_fn)
    print('\nBuilding model...')
    if args.task_type == 'classification':
        num_outputs = 2 if num_tasks == 1 else num_tasks
    else:
        num_outputs = num_tasks
    backbone = MultiModalMoleculeModel(roberta_path=args.roberta_path if args.use_1d else None, gat_checkpoint=args.gat_checkpoint if args.use_2d else None, node_feature_dim_2d=graph_2d_list[0]['x'].shape[1] if args.use_2d else 130, gat_hidden_dim=256, gat_output_dim=256, se3_checkpoint=args.se3_checkpoint if args.use_3d else None, n_atom_types=100, se3_dim=64, se3_output_dim=256, fusion_dim=args.fusion_dim, fusion_type=args.fusion_type, use_1d=args.use_1d, use_2d=args.use_2d, use_3d=args.use_3d, freeze_pretrained=args.freeze_pretrained)
    model = MultiModalPropertyPredictor(backbone, num_classes=num_outputs, task_type=args.task_type, dropout=args.dropout).to(device)
    trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(f'Trainable parameters: {trainable:,}')
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    modal_names = backbone.modal_names
    if args.task_type == 'classification':
        history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_metric': [], 'weights': []}
        if num_tasks > 1:
            history['train_auc'] = []
            history['val_auc'] = []
        best_metric = 0
        is_better = lambda new, old: new > old
        main_metric = 'roc_auc'
    else:
        history = {'train_loss': [], 'train_rmse': [], 'val_rmse': [], 'val_metric': [], 'weights': []}
        best_metric = float('inf')
        is_better = lambda new, old: new < old
        main_metric = 'rmse'
    print('\n' + '=' * 60)
    print('Starting training')
    print('=' * 60)
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        loss, train_metrics, weights = train_epoch(model, train_loader, optimizer, scheduler, device, args.task_type, num_tasks)
        if args.task_type == 'classification':
            if num_tasks == 1:
                print(f"Train - Loss: {loss:.4f}, Acc: {train_metrics.get('accuracy', 0):.4f}")
            else:
                print(f"Train - Loss: {loss:.4f}, AUC: {train_metrics.get('roc_auc', 0):.4f}")
        else:
            print(f"Train - Loss: {loss:.4f}, RMSE: {train_metrics.get('rmse', 0):.4f}")
        print(f'Modality weights: {dict(zip(modal_names, weights.round(3)))}')
        val_metrics, val_weights = evaluate(model, val_loader, device, args.task_type, num_tasks)
        if args.task_type == 'classification':
            print(f"Val - Acc: {val_metrics.get('accuracy', 'N/A')}, AUC: {val_metrics.get('roc_auc', 0):.4f}")
        else:
            print(f"Val - RMSE: {val_metrics.get('rmse', 0):.4f}, MAE: {val_metrics.get('mae', 0):.4f}")
        history['train_loss'].append(loss)
        history['weights'].append(weights.tolist())
        if args.task_type == 'classification':
            if num_tasks == 1:
                history['train_acc'].append(train_metrics.get('accuracy', 0))
                history['val_acc'].append(val_metrics.get('accuracy', 0))
            history['val_metric'].append(val_metrics.get('roc_auc', 0))
        else:
            history['train_rmse'].append(train_metrics.get('rmse', 0))
            history['val_rmse'].append(val_metrics.get('rmse', 0))
            history['val_metric'].append(val_metrics.get('rmse', float('inf')))
        current_metric = val_metrics.get(main_metric, 0 if args.task_type == 'classification' else float('inf'))
        if is_better(current_metric, best_metric):
            best_metric = current_metric
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_metrics': val_metrics, 'task_type': args.task_type, 'num_tasks': num_tasks, 'split_type': args.split_type}, os.path.join(args.output_dir, 'best_model.pt'))
            print(f'Saved best model ({main_metric.upper()}: {best_metric:.4f})')
    print('\nEvaluating best checkpoint...')
    ckpt = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(ckpt['model_state_dict'])
    test_metrics, test_weights = evaluate(model, test_loader, device, args.task_type, num_tasks)
    print('\n' + '=' * 60)
    print('Test results')
    print('=' * 60)
    print(f'Split type: {args.split_type.upper()}')
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f'{k.upper()}: {v:.4f}')
        else:
            print(f'{k.upper()}: {v}')
    print(f'\nModality weights: {dict(zip(modal_names, test_weights.round(3)))}')
    results = {'task_type': args.task_type, 'num_tasks': num_tasks, 'split_type': args.split_type, 'split_ratios': {'train': args.train_ratio, 'val': args.val_ratio, 'test': args.test_ratio}, 'split_counts': {'train': len(train_idx), 'val': len(val_idx), 'test': len(test_idx)}, 'test_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in test_metrics.items()}, 'test_weights': dict(zip(modal_names, test_weights.tolist())), f'best_val_{main_metric}': float(best_metric), 'history': history}
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    plot_curves(history, args.task_type, os.path.join(args.output_dir, 'curves.png'))
    plot_weights(history['weights'], modal_names, os.path.join(args.output_dir, 'weights.png'))
    print(f'\nResults saved to: {args.output_dir}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multimodal molecular property prediction', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='Example: python down.py --preprocessed_dir ./data/bbbp/process --task_type classification --split_type scaffold')
    parser.add_argument('--preprocessed_dir', type=str, required=True, help='Directory generated by data_process.py')
    parser.add_argument('--output_dir', type=str, default=None, help='output directory')
    parser.add_argument('--task_type', type=str, default=None, choices=['classification', 'regression'], help='Task type; inferred from config.json when omitted')
    parser.add_argument('--split_type', type=str, default='random', choices=['random', 'scaffold'], help='Dataset split strategy')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='validation split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='test split ratio')
    parser.add_argument('--stratify', action='store_true', default=True, help='Use stratified random split for single-task classification')
    parser.add_argument('--no_stratify', action='store_true', help='Disable stratified random split')
    parser.add_argument('--balanced_scaffold', action='store_true', help='Reserved flag for balanced scaffold split')
    parser.add_argument('--roberta_path', default='./PubChem10M_SMILES_BPE_450k')
    parser.add_argument('--gat_checkpoint', default='./pretrain/check_all/2d_gat/best_model.pt')
    parser.add_argument('--se3_checkpoint', default='./pretrain/check_all/egnn_student/student_best.pt')
    parser.add_argument('--fusion_dim', type=int, default=512)
    parser.add_argument('--fusion_type', default='cross', choices=['cross', 'gated'])
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use_1d', action='store_true', default=True)
    parser.add_argument('--use_2d', action='store_true', default=True)
    parser.add_argument('--use_3d', action='store_true', default=True)
    parser.add_argument('--no_1d', action='store_true', help='disable 1D modality')
    parser.add_argument('--no_2d', action='store_true', help='disable 2D modality')
    parser.add_argument('--no_3d', action='store_true', help='disable 3D modality')
    parser.add_argument('--freeze_pretrained', action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=2e-05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.no_1d:
        args.use_1d = False
    if args.no_2d:
        args.use_2d = False
    if args.no_3d:
        args.use_3d = False
    if args.no_stratify:
        args.stratify = False
    if args.output_dir is None:
        dir_name = os.path.basename(args.preprocessed_dir.rstrip('/'))
        args.output_dir = f'./output/{dir_name}_{args.split_type}'
    main(args)
