"""
Molecular data preprocessing.

Outputs:
  smiles.pkl
  labels.pkl
  2d_graphs.pkl
  3d_conformers.pkl
  metadata.pkl
  config.json
"""
import argparse
import json
import os
import pickle
import time
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
warnings.filterwarnings('ignore')

class MolecularDataPreprocessor:

    def __init__(self, csv_path, smiles_col, target_cols, task_type, output_dir, timeout=0.5, generate_3d=True):
        self.csv_path = csv_path
        self.smiles_col = smiles_col
        self.target_cols = [target_cols] if isinstance(target_cols, str) else target_cols
        self.task_type = task_type
        self.output_dir = output_dir
        self.timeout = timeout
        self.generate_3d = generate_3d
        os.makedirs(output_dir, exist_ok=True)
        self.atom_features = {'atomic_num': list(range(1, 119)), 'degree': [0, 1, 2, 3, 4, 5], 'formal_charge': [-2, -1, 0, 1, 2], 'chiral_tag': [0, 1, 2, 3], 'num_Hs': [0, 1, 2, 3, 4], 'hybridization': [0, 1, 2, 3, 4, 5]}
        self.bond_features = {'bond_type': [1, 2, 3, 12], 'is_conjugated': [0, 1], 'is_in_ring': [0, 1]}

    @staticmethod
    def one_hot_encoding(value, choices):
        encoding = [0] * (len(choices) + 1)
        index = choices.index(value) if value in choices else -1
        encoding[index] = 1
        return encoding

    @staticmethod
    def get_atom_features(atom, atom_features_dict):
        features = []
        features += MolecularDataPreprocessor.one_hot_encoding(atom.GetAtomicNum(), atom_features_dict['atomic_num'])
        features += MolecularDataPreprocessor.one_hot_encoding(atom.GetDegree(), atom_features_dict['degree'])
        features += MolecularDataPreprocessor.one_hot_encoding(atom.GetFormalCharge(), atom_features_dict['formal_charge'])
        features += MolecularDataPreprocessor.one_hot_encoding(int(atom.GetChiralTag()), atom_features_dict['chiral_tag'])
        features += MolecularDataPreprocessor.one_hot_encoding(atom.GetTotalNumHs(), atom_features_dict['num_Hs'])
        features += MolecularDataPreprocessor.one_hot_encoding(int(atom.GetHybridization()), atom_features_dict['hybridization'])
        features.append(int(atom.GetIsAromatic()))
        return features

    @staticmethod
    def get_bond_features(bond, bond_features_dict):
        features = []
        features += MolecularDataPreprocessor.one_hot_encoding(int(bond.GetBondType()), bond_features_dict['bond_type'])
        features.append(int(bond.GetIsConjugated()))
        features.append(int(bond.IsInRing()))
        return features

    def process_dataset(self, max_samples=None, num_workers=None):
        print('=' * 60)
        print('Molecular data preprocessing')
        print('=' * 60)
        print(f'CSV file: {self.csv_path}')
        print(f'SMILES column: {self.smiles_col}')
        print(f'Target columns: {self.target_cols}')
        print(f'Task type: {self.task_type}')
        print(f'Output dir: {self.output_dir}')
        print(f'Timeout: {self.timeout}s')
        print(f'Generate 3D: {self.generate_3d}')
        df = pd.read_csv(self.csv_path)
        print(f'Total samples: {len(df)}')
        if self.smiles_col not in df.columns:
            raise ValueError(f'SMILES column {self.smiles_col!r} not found. Columns: {list(df.columns)}')
        for col in self.target_cols:
            if col not in df.columns:
                raise ValueError(f'Target column {col!r} not found. Columns: {list(df.columns)}')
        if self.task_type == 'classification':
            for col in self.target_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df.loc[df[col] == -1, col] = np.nan
        if max_samples:
            df = df.head(max_samples)
            print(f'Using first {max_samples} samples')
        if num_workers is None:
            num_workers = min(cpu_count(), 16)
        print(f'Workers: {num_workers}')
        tasks = []
        for idx, row in df.iterrows():
            if len(self.target_cols) == 1:
                label = row[self.target_cols[0]]
            else:
                label = [row[col] for col in self.target_cols]
            tasks.append({'idx': idx, 'smiles': row[self.smiles_col], 'label': label})
        with Pool(processes=num_workers) as pool:
            process_func = partial(process_single_molecule_static, timeout=self.timeout, atom_features=self.atom_features, bond_features=self.bond_features, generate_3d=self.generate_3d)
            results = list(tqdm(pool.imap(process_func, tasks), total=len(tasks), desc='Processing'))
        smiles_list, labels_list = ([], [])
        graph_2d_list, conformer_3d_list, metadata_list = ([], [], [])
        failed_2d = failed_3d = timeout_count = error_count = 0
        for result in results:
            status = result['status']
            if status == 'timeout':
                timeout_count += 1
            elif status == 'failed_2d':
                failed_2d += 1
            elif status == 'failed_3d':
                failed_3d += 1
            elif status == 'error':
                error_count += 1
            elif status == 'success':
                smiles_list.append(result['smiles'])
                labels_list.append(result['label'])
                graph_2d_list.append(result['graph_2d'])
                if result.get('conformer_3d') is not None:
                    conformer_3d_list.append(result['conformer_3d'])
                metadata_list.append(result['metadata'])
        self._save_results(smiles_list, labels_list, graph_2d_list, conformer_3d_list, metadata_list, df, failed_2d, failed_3d, timeout_count, error_count)
        return (smiles_list, labels_list, graph_2d_list, conformer_3d_list, metadata_list)

    def _save_results(self, smiles_list, labels_list, graph_2d_list, conformer_3d_list, metadata_list, df, failed_2d, failed_3d, timeout_count, error_count):
        print('=' * 60)
        print('Preprocessing summary')
        print('=' * 60)
        print(f'Input samples: {len(df)}')
        print(f'Success: {len(metadata_list)}')
        print(f'2D failed: {failed_2d}')
        print(f'3D failed: {failed_3d}')
        print(f'Timeout: {timeout_count}')
        print(f'Other errors: {error_count}')
        for graph in graph_2d_list:
            graph['x'] = torch.tensor(graph['x'], dtype=torch.float32)
            graph['edge_index'] = torch.tensor(graph['edge_index'], dtype=torch.long)
            graph['edge_attr'] = torch.tensor(graph['edge_attr'], dtype=torch.float32)
            if graph['edge_attr'].numel() == 0:
                graph['edge_attr'] = graph['edge_attr'].reshape(0, 7)
        for conf in conformer_3d_list:
            conf['x'] = torch.tensor(conf['x'], dtype=torch.float32)
            conf['pos'] = torch.tensor(conf['pos'], dtype=torch.float32)
            conf['edge_index'] = torch.tensor(conf['edge_index'], dtype=torch.long)
            conf['atom_types'] = torch.tensor(conf['atom_types'], dtype=torch.long)
        save_pickle(os.path.join(self.output_dir, 'smiles.pkl'), smiles_list)
        save_pickle(os.path.join(self.output_dir, 'labels.pkl'), labels_list)
        save_pickle(os.path.join(self.output_dir, '2d_graphs.pkl'), graph_2d_list)
        save_pickle(os.path.join(self.output_dir, '3d_conformers.pkl'), conformer_3d_list)
        save_pickle(os.path.join(self.output_dir, 'metadata.pkl'), metadata_list)
        label_stats = self._label_statistics(labels_list)
        config = {'csv_path': self.csv_path, 'smiles_col': self.smiles_col, 'target_cols': self.target_cols, 'task_type': self.task_type, 'num_tasks': len(self.target_cols), 'has_3d': self.generate_3d, 'total_molecules': len(metadata_list), 'avg_num_atoms': float(np.mean([g['num_nodes'] for g in graph_2d_list])) if graph_2d_list else 0.0, 'avg_num_edges': float(np.mean([g['edge_index'].shape[1] for g in graph_2d_list])) if graph_2d_list else 0.0, 'node_feature_dim': int(graph_2d_list[0]['x'].shape[1]) if graph_2d_list else 0, 'edge_feature_dim': int(graph_2d_list[0]['edge_attr'].shape[1]) if graph_2d_list else 0, 'label_statistics': label_stats, 'timeout_count': timeout_count, 'failed_2d': failed_2d, 'failed_3d': failed_3d}
        with open(os.path.join(self.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print('Saved files:')
        print(f'  SMILES: {len(smiles_list)}')
        print(f'  labels: {len(labels_list)}')
        print(f'  2D graphs: {len(graph_2d_list)}')
        print(f'  3D conformers: {len(conformer_3d_list)}')
        print(f"  config: {os.path.join(self.output_dir, 'config.json')}")

    def _label_statistics(self, labels_list):
        labels = np.array(labels_list)
        if self.task_type == 'classification':
            if len(self.target_cols) == 1:
                valid = ~np.isnan(labels)
                return {'positive': int(np.sum(labels[valid] == 1)), 'negative': int(np.sum(labels[valid] == 0)), 'missing': int(np.sum(~valid))}
            stats = {}
            for i, col in enumerate(self.target_cols):
                col_labels = labels[:, i]
                valid = ~np.isnan(col_labels)
                stats[col] = {'positive': int(np.sum(col_labels[valid] == 1)), 'negative': int(np.sum(col_labels[valid] == 0)), 'missing': int(np.sum(~valid))}
            return stats
        if len(self.target_cols) == 1:
            return {'mean': float(np.nanmean(labels)), 'std': float(np.nanstd(labels)), 'min': float(np.nanmin(labels)), 'max': float(np.nanmax(labels))}
        stats = {}
        for i, col in enumerate(self.target_cols):
            col_labels = labels[:, i]
            stats[col] = {'mean': float(np.nanmean(col_labels)), 'std': float(np.nanstd(col_labels)), 'min': float(np.nanmin(col_labels)), 'max': float(np.nanmax(col_labels))}
        return stats

def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def process_single_molecule_static(task_data, timeout, atom_features, bond_features, generate_3d=True):
    idx = task_data['idx']
    smiles = task_data['smiles']
    label = task_data['label']
    start_time = time.time()
    try:
        if pd.isna(smiles) or str(smiles).strip() == '':
            return {'status': 'failed_2d', 'idx': idx}
        smiles = str(smiles).strip()
        graph_2d = smiles_to_2d_graph_static(smiles, atom_features, bond_features)
        if time.time() - start_time > timeout:
            return {'status': 'timeout', 'idx': idx}
        if graph_2d is None:
            return {'status': 'failed_2d', 'idx': idx}
        conformer_3d = None
        if generate_3d:
            conformer_3d = smiles_to_3d_conformer_static(smiles, atom_features, bond_features)
            if time.time() - start_time > timeout:
                return {'status': 'timeout', 'idx': idx}
            if conformer_3d is None:
                return {'status': 'failed_3d', 'idx': idx}
        return {'status': 'success', 'smiles': smiles, 'label': label, 'graph_2d': graph_2d, 'conformer_3d': conformer_3d, 'metadata': {'idx': idx, 'smiles': smiles, 'label': label}}
    except Exception as exc:
        return {'status': 'error', 'idx': idx, 'error': str(exc)}

def smiles_to_2d_graph_static(smiles, atom_features_dict, bond_features_dict):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_feature_list = [MolecularDataPreprocessor.get_atom_features(atom, atom_features_dict) for atom in mol.GetAtoms()]
    edge_indices, edge_attrs = ([], [])
    for bond in mol.GetBonds():
        i, j = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        edge_indices.extend([[i, j], [j, i]])
        bond_feature = MolecularDataPreprocessor.get_bond_features(bond, bond_features_dict)
        edge_attrs.extend([bond_feature, bond_feature])
    edge_index = np.array(edge_indices, dtype=np.int64).T.tolist() if edge_indices else [[], []]
    return {'x': atom_feature_list, 'edge_index': edge_index, 'edge_attr': edge_attrs, 'num_nodes': mol.GetNumAtoms()}

def smiles_to_3d_conformer_static(smiles, atom_features_dict, bond_features_dict, num_conformers=1, random_seed=42):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 0
    try:
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
        if len(conf_ids) == 0:
            return None
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
        except Exception:
            try:
                AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=0)
            except Exception:
                pass
        conf = mol.GetConformer(int(conf_ids[0]))
        positions = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            positions.append([pos.x, pos.y, pos.z])
        mol_no_h = Chem.RemoveHs(mol)
        atom_feature_list, atom_types_list = ([], [])
        for atom in mol_no_h.GetAtoms():
            atom_feature_list.append(MolecularDataPreprocessor.get_atom_features(atom, atom_features_dict))
            atom_types_list.append(atom.GetAtomicNum())
        edge_indices = []
        for bond in mol_no_h.GetBonds():
            i, j = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            edge_indices.extend([[i, j], [j, i]])
        heavy_atom_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
        pos_no_h = np.array(positions, dtype=np.float32)[heavy_atom_indices]
        edge_index = np.array(edge_indices, dtype=np.int64).T.tolist() if edge_indices else [[], []]
        return {'x': atom_feature_list, 'pos': pos_no_h.tolist(), 'edge_index': edge_index, 'atom_types': atom_types_list, 'num_nodes': mol_no_h.GetNumAtoms()}
    except Exception:
        return None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Molecular data preprocessing script', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='Example: python data_process.py --csv_path data.csv --smiles_col smiles --target_col all --task_type classification')
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--smiles_col', type=str, default='smiles')
    parser.add_argument('--target_col', type=str, required=True)
    parser.add_argument('--task_type', type=str, required=True, choices=['classification', 'regression'])
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--timeout', type=float, default=0.5)
    parser.add_argument('--no_3d', action='store_true', help='skip 3D conformer generation')
    args = parser.parse_args()
    if args.target_col.strip().lower() in {'all', 'auto', '*'}:
        header = pd.read_csv(args.csv_path, nrows=0)
        target_cols = [col for col in header.columns if col != args.smiles_col]
    else:
        target_cols = [col.strip() for col in args.target_col.split(',')]
    if args.output_dir is None:
        base_name = os.path.splitext(os.path.basename(args.csv_path))[0]
        args.output_dir = f'./data/{base_name}_multimodal'
    preprocessor = MolecularDataPreprocessor(csv_path=args.csv_path, smiles_col=args.smiles_col, target_cols=target_cols, task_type=args.task_type, output_dir=args.output_dir, timeout=args.timeout, generate_3d=not args.no_3d)
    smiles_list, labels_list, graphs_2d, conformers_3d, metadata = preprocessor.process_dataset(max_samples=args.max_samples, num_workers=args.num_workers)
    print('=' * 60)
    print('Preprocessing complete')
    print('=' * 60)
    print(f'SMILES: {len(smiles_list)}')
    print(f'Labels: {len(labels_list)}')
    print(f'2D graphs: {len(graphs_2d)}')
    print(f'3D conformers: {len(conformers_3d)}')
