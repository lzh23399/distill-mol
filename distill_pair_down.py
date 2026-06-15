"""
Pair-task training with Distill-Mol ligand encoder.

Supported tasks:
  DTA regression: SMILES/ligand_smiles + Protein/protein_sequence -> affinity/Y
  DTI classification: SMILES + Protein -> Y

Examples:
  python distill_pair_down.py --task dta --csv_path ./data/Protein_Ligand/davis/davis.csv     --protein_model_path ./esm2_t33_650M_UR50D --roberta_path ./PubChem10M_SMILES_BPE_450k     --output_dir ./output/pair/davis_distill --epochs 20

  python distill_pair_down.py --task dti --csv_path ./data/DTI/biosnap.csv     --protein_model_path ./esm2_t33_650M_UR50D --roberta_path ./PubChem10M_SMILES_BPE_450k     --output_dir ./output/pair/biosnap_distill --epochs 10
"""
import argparse
import json
import math
import os
import pickle
import sys
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from tqdm import tqdm
from transformers import EsmModel, EsmTokenizer, RobertaTokenizer, get_linear_schedule_with_warmup
sys.path.append('.')
from data.data_process import MolecularDataPreprocessor, process_single_molecule_static
from fusion_model import MultiModalMoleculeModel

def _resolve_columns(df, task, smiles_col, protein_col, label_col):
    if smiles_col is None:
        for candidate in ['ligand_smiles', 'SMILES', 'smiles', 'compound_iso_smiles']:
            if candidate in df.columns:
                smiles_col = candidate
                break
    if protein_col is None:
        for candidate in ['protein_sequence', 'Protein', 'target_sequence', 'sequence']:
            if candidate in df.columns:
                protein_col = candidate
                break
    if label_col is None:
        for candidate in ['affinity', 'Y', 'label']:
            if candidate in df.columns:
                label_col = candidate
                break
    missing = [name for name, value in [('smiles_col', smiles_col), ('protein_col', protein_col), ('label_col', label_col)] if value is None or value not in df.columns]
    if missing:
        raise ValueError(f'Cannot resolve columns {missing}. Available columns: {list(df.columns)}')
    print(f'Resolved columns: smiles={smiles_col}, protein={protein_col}, label={label_col}, task={task}')
    return (smiles_col, protein_col, label_col)

def _tensorize_graph(graph):
    graph = dict(graph)
    graph['x'] = torch.tensor(graph['x'], dtype=torch.float32)
    graph['edge_index'] = torch.tensor(graph['edge_index'], dtype=torch.long)
    graph['edge_attr'] = torch.tensor(graph['edge_attr'], dtype=torch.float32)
    if graph['edge_attr'].numel() == 0:
        graph['edge_attr'] = graph['edge_attr'].reshape(0, 7)
    return graph

def _tensorize_conformer(conf):
    conf = dict(conf)
    conf['x'] = torch.tensor(conf['x'], dtype=torch.float32)
    conf['pos'] = torch.tensor(conf['pos'], dtype=torch.float32).reshape(-1, 3)
    conf['edge_index'] = torch.tensor(conf['edge_index'], dtype=torch.long)
    conf['atom_types'] = torch.tensor(conf['atom_types'], dtype=torch.long)
    return conf

def _valid_conformer(conf):
    if conf is None:
        return False
    pos = conf.get('pos')
    atom_types = conf.get('atom_types')
    if pos is None or atom_types is None:
        return False
    if not torch.is_tensor(pos):
        pos = torch.tensor(pos)
    if not torch.is_tensor(atom_types):
        atom_types = torch.tensor(atom_types)
    return pos.numel() > 0 and pos.reshape(-1, 3).shape[0] > 0 and (atom_types.numel() > 0)

def build_ligand_cache(smiles_values, cache_dir, timeout=5.0, num_workers=None, no_3d=False):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'ligand_cache.pkl')
    meta_path = os.path.join(cache_dir, 'ligand_cache_meta.json')
    if os.path.exists(cache_path):
        print(f'Loading ligand cache: {cache_path}')
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        print(f'Cached ligands: {len(cache)}')
        return cache
    unique_smiles = sorted({str(s).strip() for s in smiles_values if pd.notna(s) and str(s).strip()})
    print(f'Building ligand cache for {len(unique_smiles)} unique SMILES')
    pre = MolecularDataPreprocessor(csv_path='', smiles_col='smiles', target_cols=['label'], task_type='regression', output_dir=cache_dir, timeout=timeout, generate_3d=not no_3d)
    tasks = [{'idx': i, 'smiles': smi, 'label': 0.0} for i, smi in enumerate(unique_smiles)]
    if num_workers is None:
        num_workers = min(cpu_count(), 8)
    process_func = partial(process_single_molecule_static, timeout=timeout, atom_features=pre.atom_features, bond_features=pre.bond_features, generate_3d=not no_3d)
    if num_workers <= 1:
        results = [process_func(task) for task in tqdm(tasks, desc='Ligand cache')]
    else:
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(process_func, tasks), total=len(tasks), desc='Ligand cache'))
    cache = {}
    failed = {}
    for result in results:
        if result.get('status') != 'success':
            failed[result.get('idx')] = result.get('status')
            continue
        smi = result['smiles']
        graph = _tensorize_graph(result['graph_2d'])
        conf = result.get('conformer_3d')
        if conf is not None:
            conf = _tensorize_conformer(conf)
            if not _valid_conformer(conf):
                conf = None
        cache[smi] = {'graph_2d': graph, 'conformer_3d': conf}
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({'unique_smiles': len(unique_smiles), 'cached': len(cache), 'failed': len(failed), 'no_3d': no_3d}, f, indent=2)
    print(f'Saved ligand cache: {cache_path}')
    print(f'Cached={len(cache)}, failed={len(failed)}')
    return cache

class PairDataset(Dataset):

    def __init__(self, df, ligand_cache, ligand_tokenizer, protein_tokenizer, smiles_col, protein_col, label_col, task, max_ligand_length=128, max_protein_length=1024, use_1d=True, use_2d=True, use_3d=True):
        self.df = df.reset_index(drop=True)
        self.ligand_cache = ligand_cache
        self.ligand_tokenizer = ligand_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.smiles_col = smiles_col
        self.protein_col = protein_col
        self.label_col = label_col
        self.task = task
        self.max_ligand_length = max_ligand_length
        self.max_protein_length = max_protein_length
        self.use_1d = use_1d
        self.use_2d = use_2d
        self.use_3d = use_3d

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = str(row[self.smiles_col]).strip()
        protein = str(row[self.protein_col]).strip()
        label = float(row[self.label_col])
        cached = self.ligand_cache[smiles]
        sample = {'smiles': smiles, 'protein': protein}
        if self.task == 'dti':
            sample['label'] = torch.tensor(int(label), dtype=torch.long)
        else:
            sample['label'] = torch.tensor(label, dtype=torch.float32)
        if self.use_1d:
            enc = self.ligand_tokenizer(smiles, max_length=self.max_ligand_length, padding='max_length', truncation=True, return_tensors='pt')
            sample['1d'] = {'input_ids': enc['input_ids'].flatten(), 'attention_mask': enc['attention_mask'].flatten()}
        if self.use_2d:
            graph = cached['graph_2d']
            sample['2d'] = Data(x=graph['x'], edge_index=graph['edge_index'], edge_attr=graph['edge_attr'])
        if self.use_3d:
            conf = cached['conformer_3d']
            if not _valid_conformer(conf):
                raise ValueError(f'Invalid 3D conformer for SMILES: {smiles}')
            sample['3d'] = {'atom_types': conf['atom_types'], 'positions': conf['pos'].reshape(-1, 3), 'num_atoms': conf['num_nodes']}
        p_enc = self.protein_tokenizer(protein, max_length=self.max_protein_length, padding='max_length', truncation=True, return_tensors='pt')
        sample['protein_input_ids'] = p_enc['input_ids'].flatten()
        sample['protein_attention_mask'] = p_enc['attention_mask'].flatten()
        return sample

def pair_collate_fn(batch):
    result = {'label': torch.stack([b['label'] for b in batch]), 'smiles': [b['smiles'] for b in batch], 'protein': [b['protein'] for b in batch], 'protein_input_ids': torch.stack([b['protein_input_ids'] for b in batch]), 'protein_attention_mask': torch.stack([b['protein_attention_mask'] for b in batch])}
    if '1d' in batch[0]:
        result['1d'] = {'input_ids': torch.stack([b['1d']['input_ids'] for b in batch]), 'attention_mask': torch.stack([b['1d']['attention_mask'] for b in batch])}
    if '2d' in batch[0]:
        result['2d'] = Batch.from_data_list([b['2d'] for b in batch])
    if '3d' in batch[0]:
        max_n = max((b['3d']['positions'].shape[0] for b in batch))
        bsz = len(batch)
        atom_types = torch.zeros(bsz, max_n, dtype=torch.long)
        positions = torch.zeros(bsz, max_n, 3)
        mask = torch.zeros(bsz, max_n)
        for i, item in enumerate(batch):
            pos = item['3d']['positions'].reshape(-1, 3)
            n = pos.shape[0]
            atom_types[i, :n] = item['3d']['atom_types'][:n]
            positions[i, :n] = pos
            mask[i, :n] = 1.0
        result['3d'] = {'atom_types': atom_types, 'positions': positions, 'mask': mask}
    return result

class DistillProteinPairModel(nn.Module):

    def __init__(self, protein_model_path, roberta_path, gat_checkpoint=None, se3_checkpoint=None, node_feature_dim_2d=130, fusion_dim=512, task='dta', use_1d=True, use_2d=True, use_3d=True, freeze_ligand=True, freeze_protein=True, dropout=0.2):
        super().__init__()
        self.task = task
        self.ligand_encoder = MultiModalMoleculeModel(roberta_path=roberta_path, gat_checkpoint=gat_checkpoint, se3_checkpoint=se3_checkpoint, node_feature_dim_2d=node_feature_dim_2d, fusion_dim=fusion_dim, fusion_type='cross', use_1d=use_1d, use_2d=use_2d, use_3d=use_3d, freeze_pretrained=freeze_ligand, dropout=dropout)
        self.protein_encoder = EsmModel.from_pretrained(protein_model_path)
        protein_dim = self.protein_encoder.config.hidden_size
        if freeze_protein:
            for p in self.protein_encoder.parameters():
                p.requires_grad = False
        self.protein_proj = nn.Sequential(nn.Linear(protein_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU(), nn.Dropout(dropout))
        out_dim = 2 if task == 'dti' else 1
        pair_dim = fusion_dim * 4
        self.head = nn.Sequential(nn.Linear(pair_dim, fusion_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fusion_dim // 2, out_dim))

    def forward(self, batch):
        ligand_batch = {}
        for key in ['1d', '2d', '3d']:
            if key in batch:
                ligand_batch[key] = batch[key]
        z_ligand = self.ligand_encoder(ligand_batch)
        protein_out = self.protein_encoder(input_ids=batch['protein_input_ids'], attention_mask=batch['protein_attention_mask'])
        z_protein = self.protein_proj(protein_out.last_hidden_state[:, 0, :])
        pair = torch.cat([z_ligand, z_protein, torch.abs(z_ligand - z_protein), z_ligand * z_protein], dim=-1)
        out = self.head(pair)
        if self.task == 'dta':
            return out.squeeze(-1)
        return out

def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if key in {'smiles', 'protein'}:
            moved[key] = value
        elif isinstance(value, dict):
            moved[key] = {k: v.to(device) for k, v in value.items()}
        elif hasattr(value, 'to'):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved

def train_one_epoch(model, loader, optimizer, scheduler, device, task):
    model.train()
    criterion = nn.CrossEntropyLoss() if task == 'dti' else nn.MSELoss()
    total_loss = 0.0
    for batch in tqdm(loader, desc='Train', leave=False):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch['label'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)

def concordance_index(y_true, y_pred):
    """Concordance index used by common DTA benchmarks."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = 0
    h_sum = 0.0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] == y_true[j]:
                continue
            n += 1
            true_order = y_true[i] > y_true[j]
            pred_order = y_pred[i] > y_pred[j]
            if pred_order == true_order:
                h_sum += 1.0
            elif y_pred[i] == y_pred[j]:
                h_sum += 0.5
    return float(h_sum / n) if n > 0 else 0.0

@torch.no_grad()
def evaluate(model, loader, device, task):
    model.eval()
    labels, preds, probs = ([], [], [])
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss() if task == 'dti' else nn.MSELoss()
    for batch in tqdm(loader, desc='Eval', leave=False):
        batch = move_batch_to_device(batch, device)
        pred = model(batch)
        loss = criterion(pred, batch['label'])
        total_loss += float(loss.item())
        if task == 'dti':
            prob = torch.softmax(pred, dim=-1)[:, 1]
            probs.extend(prob.detach().cpu().numpy().tolist())
            preds.extend(torch.argmax(pred, dim=-1).detach().cpu().numpy().tolist())
            labels.extend(batch['label'].detach().cpu().numpy().tolist())
        else:
            preds.extend(pred.detach().cpu().numpy().tolist())
            labels.extend(batch['label'].detach().cpu().numpy().tolist())
    labels_np = np.asarray(labels)
    preds_np = np.asarray(preds)
    metrics = {'loss': total_loss / max(len(loader), 1)}
    if task == 'dti':
        probs_np = np.asarray(probs)
        metrics.update({'accuracy': float(accuracy_score(labels_np, preds_np)), 'precision': float(precision_score(labels_np, preds_np, zero_division=0)), 'recall': float(recall_score(labels_np, preds_np, zero_division=0)), 'f1': float(f1_score(labels_np, preds_np, zero_division=0)), 'roc_auc': float(roc_auc_score(labels_np, probs_np)) if len(set(labels_np.tolist())) > 1 else 0.0})
    else:
        mse = mean_squared_error(labels_np, preds_np)
        metrics.update({'mse': float(mse), 'rmse': float(math.sqrt(mse)), 'mae': float(mean_absolute_error(labels_np, preds_np)), 'ci': concordance_index(labels_np, preds_np), 'r2': float(r2_score(labels_np, preds_np)), 'pearson': float(pearsonr(labels_np, preds_np)[0]) if len(labels_np) > 1 else 0.0, 'spearman': float(spearmanr(labels_np, preds_np)[0]) if len(labels_np) > 1 else 0.0})
    return metrics

def prepare_dataframe(args):
    df = pd.read_csv(args.csv_path)
    smiles_col, protein_col, label_col = _resolve_columns(df, args.task, args.smiles_col, args.protein_col, args.label_col)
    df = df[[smiles_col, protein_col, label_col]].copy()
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df[protein_col] = df[protein_col].astype(str).str.strip()
    df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
    df = df.dropna(subset=[smiles_col, protein_col, label_col])
    df = df[(df[smiles_col] != '') & (df[protein_col] != '')]
    if args.task == 'dti':
        df = df[df[label_col].isin([0, 1, 0.0, 1.0])]
        df[label_col] = df[label_col].astype(int)
    if args.max_samples:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=args.seed).reset_index(drop=True)
    print(f'Usable rows: {len(df)}')
    print(f'Unique ligands: {df[smiles_col].nunique()}, unique proteins: {df[protein_col].nunique()}')
    return (df, smiles_col, protein_col, label_col)

def split_dataframe(df, label_col, task, seed):
    stratify = df[label_col] if task == 'dti' and df[label_col].nunique() == 2 else None
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=stratify)
    stratify_temp = temp_df[label_col] if task == 'dti' and temp_df[label_col].nunique() == 2 else None
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=stratify_temp)
    return (train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True))

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'Device: {device}')
    df, smiles_col, protein_col, label_col = prepare_dataframe(args)
    ligand_cache = build_ligand_cache(df[smiles_col].tolist(), cache_dir=os.path.join(args.output_dir, 'ligand_cache'), timeout=args.timeout, num_workers=args.num_workers, no_3d=args.no_3d)
    df = df[df[smiles_col].isin(ligand_cache.keys())].reset_index(drop=True)
    if not args.no_3d:
        df = df[df[smiles_col].map(lambda s: _valid_conformer(ligand_cache[s]['conformer_3d']))].reset_index(drop=True)
    print(f'Rows after ligand cache filtering: {len(df)}')
    if args.train_csv and args.val_csv and args.test_csv:
        train_df = pd.read_csv(args.train_csv)
        val_df = pd.read_csv(args.val_csv)
        test_df = pd.read_csv(args.test_csv)
        smiles_col, protein_col, label_col = _resolve_columns(train_df, args.task, args.smiles_col, args.protein_col, args.label_col)
    else:
        train_df, val_df, test_df = split_dataframe(df, label_col, args.task, args.seed)
    print(f'Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')
    ligand_tokenizer = RobertaTokenizer.from_pretrained(args.roberta_path)
    protein_tokenizer = EsmTokenizer.from_pretrained(args.protein_model_path)
    common_ds = dict(ligand_cache=ligand_cache, ligand_tokenizer=ligand_tokenizer, protein_tokenizer=protein_tokenizer, smiles_col=smiles_col, protein_col=protein_col, label_col=label_col, task=args.task, max_ligand_length=args.max_ligand_length, max_protein_length=args.max_protein_length, use_1d=not args.no_1d, use_2d=not args.no_2d, use_3d=not args.no_3d)
    train_ds = PairDataset(train_df, **common_ds)
    val_ds = PairDataset(val_df, **common_ds)
    test_ds = PairDataset(test_df, **common_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pair_collate_fn, num_workers=0, drop_last=args.batch_size > 1)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pair_collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pair_collate_fn, num_workers=0)
    node_dim = next(iter(ligand_cache.values()))['graph_2d']['x'].shape[1]
    model = DistillProteinPairModel(protein_model_path=args.protein_model_path, roberta_path=args.roberta_path, gat_checkpoint=args.gat_checkpoint, se3_checkpoint=args.se3_checkpoint, node_feature_dim_2d=node_dim, fusion_dim=args.fusion_dim, task=args.task, use_1d=not args.no_1d, use_2d=not args.no_2d, use_3d=not args.no_3d, freeze_ligand=not args.unfreeze_ligand, freeze_protein=not args.unfreeze_protein, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_ratio * total_steps), num_training_steps=total_steps)
    best_metric = -float('inf') if args.task == 'dti' else float('inf')
    best_path = os.path.join(args.output_dir, 'best_model.pt')
    history = []
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, args.task)
        val_metrics = evaluate(model, val_loader, device, args.task)
        print(f'Train loss: {train_loss:.4f}')
        print(f'Val metrics: {val_metrics}')
        current = val_metrics['roc_auc'] if args.task == 'dti' else val_metrics['mse']
        is_better = current > best_metric if args.task == 'dti' else current < best_metric
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val': val_metrics})
        if is_better:
            best_metric = current
            torch.save({'model_state_dict': model.state_dict(), 'args': vars(args)}, best_path)
            print(f'Saved best model: {best_path}')
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(model, test_loader, device, args.task)
    print('\n' + '=' * 60)
    print('Test metrics')
    print('=' * 60)
    for key, value in test_metrics.items():
        print(f'{key}: {value:.6f}')
    with open(os.path.join(args.output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump({'task': args.task, 'csv_path': args.csv_path, 'columns': {'smiles': smiles_col, 'protein': protein_col, 'label': label_col}, 'test_metrics': test_metrics, 'history': history, 'args': vars(args)}, f, indent=2, ensure_ascii=False)
    print(f'Saved results to {args.output_dir}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distill-Mol pair-task training for DTA/DTI')
    parser.add_argument('--task', choices=['dta', 'dti'], required=True)
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--smiles_col', default=None)
    parser.add_argument('--protein_col', default=None)
    parser.add_argument('--label_col', default=None)
    parser.add_argument('--train_csv', default=None)
    parser.add_argument('--val_csv', default=None)
    parser.add_argument('--test_csv', default=None)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--protein_model_path', default='./esm2_t33_650M_UR50D')
    parser.add_argument('--roberta_path', default='./PubChem10M_SMILES_BPE_450k')
    parser.add_argument('--gat_checkpoint', default=None)
    parser.add_argument('--se3_checkpoint', default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--fusion_dim', type=int, default=512)
    parser.add_argument('--max_ligand_length', type=int, default=128)
    parser.add_argument('--max_protein_length', type=int, default=1024)
    parser.add_argument('--timeout', type=float, default=5.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--no_1d', action='store_true')
    parser.add_argument('--no_2d', action='store_true')
    parser.add_argument('--no_3d', action='store_true')
    parser.add_argument('--unfreeze_ligand', action='store_true')
    parser.add_argument('--unfreeze_protein', action='store_true')
    main(parser.parse_args())
