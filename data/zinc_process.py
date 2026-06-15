"""Distill-Mol module."""
import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
import torch
from torch_geometric.data import Data, Dataset
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from collections import Counter
warnings.filterwarnings('ignore')

class ZINCMultiModalPreprocessor:
    """ZINCMultiModalPreprocessor implementation."""

    def __init__(self, zinc_csv_path, output_dir='./data/zinc_multimodal', timeout=0.5):
        self.zinc_csv_path = zinc_csv_path
        self.output_dir = output_dir
        self.timeout = timeout
        os.makedirs(output_dir, exist_ok=True)
        self.atom_features = {'atomic_num': [5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53], 'degree': [0, 1, 2, 3, 4, 5], 'formal_charge': [-2, -1, 0, 1, 2], 'chiral_tag': [0, 1, 2, 3], 'num_Hs': [0, 1, 2, 3, 4], 'hybridization': [0, 1, 2, 3, 4, 5]}
        self.bond_features = {'bond_type': [1, 2, 3, 12], 'is_conjugated': [0, 1], 'is_in_ring': [0, 1]}
        self.atomic_num_to_symbol = {1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
        self.atom_type_counter = Counter()
        self.processing_stats = {'start_time': None, 'end_time': None, 'total_atoms': 0, 'total_bonds': 0}

    def print_config(self):
        """print_config helper."""
        print('\n' + '=' * 80)
        print('ZINC preprocessing configuration')
        print('=' * 80)
        print('\nAtom feature configuration:')
        print(f"  Supported atom types: {len(self.atom_features['atomic_num'])}")
        print(f"  Atomic numbers: {self.atom_features['atomic_num']}")
        elements = [self.atomic_num_to_symbol.get(num, f'?{num}') for num in self.atom_features['atomic_num']]
        print(f"  Elements: {', '.join(elements)}")
        node_feat_dim = len(self.atom_features['atomic_num']) + 1 + len(self.atom_features['degree']) + 1 + len(self.atom_features['formal_charge']) + 1 + len(self.atom_features['chiral_tag']) + 1 + len(self.atom_features['num_Hs']) + 1 + len(self.atom_features['hybridization']) + 1 + 1
        edge_feat_dim = len(self.bond_features['bond_type']) + 1 + 1 + 1
        print('\nFeature dimensions:')
        print(f'  Node feature dim: {node_feat_dim}')
        print(f"    - atomic_num: {len(self.atom_features['atomic_num']) + 1}")
        print(f"    - degree: {len(self.atom_features['degree']) + 1}")
        print(f"    - formal_charge: {len(self.atom_features['formal_charge']) + 1}")
        print(f"    - chiral_tag: {len(self.atom_features['chiral_tag']) + 1}")
        print(f"    - num_Hs: {len(self.atom_features['num_Hs']) + 1}")
        print(f"    - hybridization: {len(self.atom_features['hybridization']) + 1}")
        print(f'    - is_aromatic: 1')
        print(f'  Edge feature dim: {edge_feat_dim}')
        original_node_dim = 119 + 7 + 6 + 5 + 6 + 7 + 1
        print('\nDimension reduction:')
        print('  Original node dim: 151')
        print(f'  Current node dim: {node_feat_dim}')
        print(f'  Saved dims: {151 - node_feat_dim} ({(151 - node_feat_dim) / 151 * 100:.1f}%)')
        print('=' * 80)

    @staticmethod
    def one_hot_encoding(value, choices):
        """one_hot_encoding helper."""
        encoding = [0] * (len(choices) + 1)
        index = choices.index(value) if value in choices else -1
        encoding[index] = 1
        return encoding

    @staticmethod
    def get_atom_features(atom, atom_features_dict):
        """get_atom_features helper."""
        features = []
        features += ZINCMultiModalPreprocessor.one_hot_encoding(atom.GetAtomicNum(), atom_features_dict['atomic_num'])
        features += ZINCMultiModalPreprocessor.one_hot_encoding(atom.GetDegree(), atom_features_dict['degree'])
        features += ZINCMultiModalPreprocessor.one_hot_encoding(atom.GetFormalCharge(), atom_features_dict['formal_charge'])
        features += ZINCMultiModalPreprocessor.one_hot_encoding(int(atom.GetChiralTag()), atom_features_dict['chiral_tag'])
        features += ZINCMultiModalPreprocessor.one_hot_encoding(atom.GetTotalNumHs(), atom_features_dict['num_Hs'])
        features += ZINCMultiModalPreprocessor.one_hot_encoding(int(atom.GetHybridization()), atom_features_dict['hybridization'])
        features.append(int(atom.GetIsAromatic()))
        return features

    @staticmethod
    def get_bond_features(bond, bond_features_dict):
        """get_bond_features helper."""
        features = []
        features += ZINCMultiModalPreprocessor.one_hot_encoding(int(bond.GetBondType()), bond_features_dict['bond_type'])
        features.append(int(bond.GetIsConjugated()))
        features.append(int(bond.IsInRing()))
        return features

    def process_zinc_dataset(self, max_samples=None, num_workers=None):
        """process_zinc_dataset helper."""
        print('=' * 80)
        print('ZINC multimodal preprocessing')
        print(f'Timeout per molecule: {self.timeout}s')
        print('=' * 80)
        self.print_config()
        self.processing_stats['start_time'] = time.time()
        df = pd.read_csv(self.zinc_csv_path)
        print('\nInput data:')
        print(f'  SMILES: {self.zinc_csv_path}')
        print(f'  Total rows: {len(df):,}')
        if max_samples:
            df = df.head(max_samples)
            print(f'  Max samples: {max_samples:,}')
        if num_workers is None:
            num_workers = min(cpu_count(), 16)
        print(f'  Workers: {num_workers}')
        tasks = []
        for idx, row in df.iterrows():
            tasks.append({'idx': idx, 'smiles': row['smiles'], 'row_dict': row.to_dict()})
        print(f"\n{'=' * 80}")
        print('Processing molecules...')
        print(f"{'=' * 80}")
        with Pool(processes=num_workers) as pool:
            process_func = partial(process_single_molecule_static, timeout=self.timeout, atom_features=self.atom_features, bond_features=self.bond_features)
            results = list(tqdm(pool.imap(process_func, tasks), total=len(tasks), desc='', ncols=100))
        graph_2d_list = []
        conformer_3d_list = []
        metadata_list = []
        failed_2d = 0
        failed_3d = 0
        timeout_count = 0
        error_count = 0
        for result in results:
            if result['status'] == 'timeout':
                timeout_count += 1
            elif result['status'] == 'failed_2d':
                failed_2d += 1
            elif result['status'] == 'failed_3d':
                failed_3d += 1
            elif result['status'] == 'error':
                error_count += 1
            elif result['status'] == 'success':
                graph_2d_list.append(result['graph_2d'])
                conformer_3d_list.append(result['conformer_3d'])
                metadata_list.append(result['metadata'])
        self.processing_stats['end_time'] = time.time()
        self._save_results(graph_2d_list, conformer_3d_list, metadata_list, df, failed_2d, failed_3d, timeout_count, error_count)
        return (graph_2d_list, conformer_3d_list, metadata_list)

    def _save_results(self, graph_2d_list, conformer_3d_list, metadata_list, df, failed_2d, failed_3d, timeout_count, error_count):
        """_save_results helper."""
        print(f"\n{'=' * 80}")
        print('Preprocessing summary')
        print(f"{'=' * 80}")
        total_input = len(df)
        total_success = len(metadata_list)
        total_failed = failed_2d + failed_3d + timeout_count + error_count
        success_rate = total_success / total_input * 100 if total_input > 0 else 0
        print('\nCounts:')
        print(f'  Input molecules: {total_input:,}')
        print(f'  Successful molecules: {total_success:,}')
        print(f'  Failed molecules: {total_failed:,}')
        print(f'     2D: {failed_2d:,}')
        print(f'     3D: {failed_3d:,}')
        print(f'     Timeout: {timeout_count:,}')
        print(f'     Other errors: {error_count:,}')
        print(f'  Success rate: {success_rate:.2f}%')
        if self.processing_stats['start_time'] and self.processing_stats['end_time']:
            elapsed = self.processing_stats['end_time'] - self.processing_stats['start_time']
            avg_time = elapsed / total_input if total_input > 0 else 0
            print('\nRuntime:')
            print(f'  Elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)')
            print(f'  Average time: {avg_time * 1000:.2f} ms/molecule')
            print(f'  Throughput: {total_input / elapsed:.1f} molecules/s')
        print(f"\n{'=' * 80}")
        print('Converting arrays to PyTorch tensors...')
        print(f"{'=' * 80}")
        atom_type_counter = Counter()
        total_atoms = 0
        total_bonds = 0
        print('\nConverting 2D graphs...')
        for i, graph in enumerate(tqdm(graph_2d_list, desc='2D', ncols=100)):
            try:
                if not isinstance(graph['x'], torch.Tensor):
                    graph['x'] = torch.tensor(graph['x'], dtype=torch.float32)
                if not isinstance(graph['edge_index'], torch.Tensor):
                    graph['edge_index'] = torch.tensor(graph['edge_index'], dtype=torch.long)
                if not isinstance(graph['edge_attr'], torch.Tensor):
                    graph['edge_attr'] = torch.tensor(graph['edge_attr'], dtype=torch.float32)
                total_atoms += graph['num_nodes']
                total_bonds += graph['edge_index'].shape[1] // 2
            except Exception as e:
                print(f'\nFailed to convert 2D graph {i}: {e}')
                raise
        print('\nConverting 3D conformers...')
        for i, conf in enumerate(tqdm(conformer_3d_list, desc='3D', ncols=100)):
            try:
                if not isinstance(conf['x'], torch.Tensor):
                    conf['x'] = torch.tensor(conf['x'], dtype=torch.float32)
                if not isinstance(conf['pos'], torch.Tensor):
                    conf['pos'] = torch.tensor(conf['pos'], dtype=torch.float32)
                if not isinstance(conf['edge_index'], torch.Tensor):
                    conf['edge_index'] = torch.tensor(conf['edge_index'], dtype=torch.long)
            except Exception as e:
                print(f'\nFailed to convert 3D conformer {i}: {e}')
                raise
        print(f"\n{'=' * 80}")
        print('Saving processed files...')
        print(f"{'=' * 80}")
        graph_2d_path = os.path.join(self.output_dir, '2d_graphs.pkl')
        conformer_3d_path = os.path.join(self.output_dir, '3d_conformers.pkl')
        metadata_path = os.path.join(self.output_dir, 'metadata.pkl')
        print(f'\nOutput directory: {self.output_dir}')
        with open(graph_2d_path, 'wb') as f:
            pickle.dump(graph_2d_list, f)
        file_size_mb = os.path.getsize(graph_2d_path) / (1024 * 1024)
        print(f'   2D: {graph_2d_path}')
        print(f'    Size: {file_size_mb:.2f} MB')
        with open(conformer_3d_path, 'wb') as f:
            pickle.dump(conformer_3d_list, f)
        file_size_mb = os.path.getsize(conformer_3d_path) / (1024 * 1024)
        print(f'   3D: {conformer_3d_path}')
        print(f'    Size: {file_size_mb:.2f} MB')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata_list, f)
        file_size_mb = os.path.getsize(metadata_path) / (1024 * 1024)
        print(f'   Metadata: {metadata_path}')
        print(f'    Size: {file_size_mb:.2f} MB')
        if len(graph_2d_list) > 0:
            self._generate_detailed_report(graph_2d_list, conformer_3d_list, metadata_list, total_input, total_success, failed_2d, failed_3d, timeout_count, error_count, total_atoms, total_bonds)

    def _generate_detailed_report(self, graph_2d_list, conformer_3d_list, metadata_list, total_input, total_success, failed_2d, failed_3d, timeout_count, error_count, total_atoms, total_bonds):
        """_generate_detailed_report helper."""
        n_atoms_list = [g['num_nodes'] for g in graph_2d_list]
        n_edges_list = [g['edge_index'].shape[1] for g in graph_2d_list]
        node_dim = graph_2d_list[0]['x'].shape[1]
        edge_dim = graph_2d_list[0]['edge_attr'].shape[1]
        stats = {'total_molecules': len(metadata_list), 'avg_num_atoms': float(np.mean(n_atoms_list)), 'median_num_atoms': float(np.median(n_atoms_list)), 'min_num_atoms': int(np.min(n_atoms_list)), 'max_num_atoms': int(np.max(n_atoms_list)), 'avg_num_edges': float(np.mean(n_edges_list)), 'node_feature_dim': int(node_dim), 'edge_feature_dim': int(edge_dim), 'total_atoms': int(total_atoms), 'total_bonds': int(total_bonds), 'timeout_count': timeout_count, 'failed_2d': failed_2d, 'failed_3d': failed_3d, 'error_count': error_count, 'success_rate': float(total_success / total_input * 100) if total_input > 0 else 0, 'atomic_num_config': self.atom_features['atomic_num']}
        if self.processing_stats['start_time'] and self.processing_stats['end_time']:
            elapsed = self.processing_stats['end_time'] - self.processing_stats['start_time']
            stats['processing_time_seconds'] = float(elapsed)
            stats['processing_speed_molecules_per_second'] = float(total_input / elapsed) if elapsed > 0 else 0
        stats_path = os.path.join(self.output_dir, 'statistics.json')
        import json
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f'   Statistics: {stats_path}')
        print(f"\n{'=' * 80}")
        print('Dataset statistics')
        print(f"{'=' * 80}")
        print('\nMolecule statistics:')
        print(f"  Molecules: {stats['total_molecules']:,}")
        print(f'  Total atoms: {total_atoms:,}')
        print(f'  Total bonds: {total_bonds:,}')
        print(f"  Average atoms: {stats['avg_num_atoms']:.2f}")
        print(f"  Median atoms: {stats['median_num_atoms']:.0f}")
        print(f"  Atom range: [{stats['min_num_atoms']}, {stats['max_num_atoms']}]")
        print(f"  Average directed edges: {stats['avg_num_edges']:.2f}")
        print('\nFeature statistics:')
        print(f'  Node feature dim: {node_dim}')
        print(f'  Edge feature dim: {edge_dim}')
        print(f"  Supported atom types: {len(self.atom_features['atomic_num'])}")
        original_node_dim = 151
        if node_dim < original_node_dim:
            saved_dim = original_node_dim - node_dim
            saved_percent = saved_dim / original_node_dim * 100
            print('\nDimension reduction:')
            print(f'    Original node dim: {original_node_dim}')
            print(f'    Current node dim: {node_dim}')
            print(f'    Saved dims: {saved_dim} ({saved_percent:.1f}%)')
            memory_saved_mb = saved_dim * total_atoms * 4 / (1024 * 1024)
            print(f'    Estimated memory saved: {memory_saved_mb:.2f} MB')
        print('\nQuality:')
        success_rate = stats['success_rate']
        if success_rate >= 95:
            quality = 'excellent'
        elif success_rate >= 90:
            quality = 'good'
        elif success_rate >= 80:
            quality = 'acceptable'
        else:
            quality = 'needs review'
        print(f'  Success rate: {success_rate:.2f}% - {quality}')
        print(f'  2D failure rate: {failed_2d / total_input * 100:.2f}%')
        print(f'  3D failure rate: {failed_3d / total_input * 100:.2f}%')
        print(f'  Timeout rate: {timeout_count / total_input * 100:.2f}%')
        self._save_text_report(stats, total_input)
        print(f"\n{'=' * 80}")
        print('ZINC preprocessing complete')
        print(f"{'=' * 80}")

    def _save_text_report(self, stats, total_input):
        """_save_text_report helper."""
        report_path = os.path.join(self.output_dir, 'processing_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('=' * 80 + '\n')
            f.write('ZINC\n')
            f.write('=' * 80 + '\n\n')
            f.write('Atom feature configuration\n')
            f.write(f"  Supported atom types: {len(self.atom_features['atomic_num'])}\n")
            f.write(f"  Atomic numbers: {self.atom_features['atomic_num']}\n")
            elements = [self.atomic_num_to_symbol.get(num, f'?{num}') for num in self.atom_features['atomic_num']]
            f.write(f"  Elements: {', '.join(elements)}\n")
            f.write(f"  Node feature dim: {stats['node_feature_dim']}\n")
            f.write(f"  Edge feature dim: {stats['edge_feature_dim']}\n\n")
            f.write('Processing summary\n')
            f.write(f'  Input molecules: {total_input:,}\n')
            f.write(f"  Successful molecules: {stats['total_molecules']:,}\n")
            f.write(f"  2D failures: {stats['failed_2d']:,}\n")
            f.write(f"  3D failures: {stats['failed_3d']:,}\n")
            f.write(f"  Timeouts: {stats['timeout_count']:,}\n")
            f.write(f"  Other errors: {stats['error_count']:,}\n")
            f.write(f"  Success rate: {stats['success_rate']:.2f}%\n\n")
            f.write('Dataset statistics\n')
            f.write(f"  Molecules: {stats['total_molecules']:,}\n")
            f.write(f"  Total atoms: {stats['total_atoms']:,}\n")
            f.write(f"  Total bonds: {stats['total_bonds']:,}\n")
            f.write(f"  Average atoms: {stats['avg_num_atoms']:.2f}\n")
            f.write(f"  Median atoms: {stats['median_num_atoms']:.0f}\n")
            f.write(f"  Atom range: [{stats['min_num_atoms']}, {stats['max_num_atoms']}]\n")
            f.write(f"  Average directed edges: {stats['avg_num_edges']:.2f}\n\n")
            if 'processing_time_seconds' in stats:
                f.write('Runtime\n')
                f.write(f"  Elapsed: {stats['processing_time_seconds']:.1f}s\n")
                f.write(f"  Throughput: {stats['processing_speed_molecules_per_second']:.1f} molecules/s\n\n")
            f.write('Feature dimensions\n')
            original_dim = 151
            if stats['node_feature_dim'] < original_dim:
                saved = original_dim - stats['node_feature_dim']
                f.write(f'  Original node dim: {original_dim}\n')
                f.write(f"  Current node dim: {stats['node_feature_dim']}\n")
                f.write(f'  Saved dims: {saved} ({saved / original_dim * 100:.1f}%)\n')
            else:
                f.write(f"  Node feature dim: {stats['node_feature_dim']}\n")
            f.write('\n' + '=' * 80 + '\n')
        print(f'   Text report: {report_path}')

def process_single_molecule_static(task_data, timeout, atom_features, bond_features):
    """process_single_molecule_static helper."""
    idx = task_data['idx']
    smiles = task_data['smiles']
    row_dict = task_data['row_dict']
    start_time = time.time()
    try:
        graph_2d = smiles_to_2d_graph_static(smiles, atom_features, bond_features)
        if time.time() - start_time > timeout:
            return {'status': 'timeout', 'idx': idx}
        if graph_2d is None:
            return {'status': 'failed_2d', 'idx': idx}
        conformer_3d = smiles_to_3d_conformer_static(smiles, atom_features, bond_features)
        if time.time() - start_time > timeout:
            return {'status': 'timeout', 'idx': idx}
        if conformer_3d is None:
            return {'status': 'failed_3d', 'idx': idx}
        metadata = {'idx': idx, 'smiles': smiles}
        if 'logP' in row_dict:
            metadata['logP'] = row_dict['logP']
        if 'mol_weight' in row_dict:
            metadata['mol_weight'] = row_dict['mol_weight']
        return {'status': 'success', 'graph_2d': graph_2d, 'conformer_3d': conformer_3d, 'metadata': metadata}
    except Exception as e:
        return {'status': 'error', 'idx': idx, 'error': str(e)}

def smiles_to_2d_graph_static(smiles, atom_features_dict, bond_features_dict):
    """smiles_to_2d_graph_static helper."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_feature_list = []
    for atom in mol.GetAtoms():
        features = ZINCMultiModalPreprocessor.get_atom_features(atom, atom_features_dict)
        atom_feature_list.append(features)
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        bond_feature = ZINCMultiModalPreprocessor.get_bond_features(bond, bond_features_dict)
        edge_attrs.append(bond_feature)
        edge_attrs.append(bond_feature)
    return {'x': atom_feature_list, 'edge_index': [list(e) for e in np.array(edge_indices, dtype=np.int64).T], 'edge_attr': edge_attrs, 'num_nodes': mol.GetNumAtoms()}

def smiles_to_3d_conformer_static(smiles, atom_features_dict, bond_features_dict, num_conformers=1, random_seed=42):
    """smiles_to_3d_conformer_static helper."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 0
    try:
        result = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
        if result == -1 or len(result) == 0:
            return None
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
        conf = mol.GetConformer(0)
        positions = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            positions.append([pos.x, pos.y, pos.z])
        mol_no_h = Chem.RemoveHs(mol)
        atom_feature_list = []
        for atom in mol_no_h.GetAtoms():
            features = ZINCMultiModalPreprocessor.get_atom_features(atom, atom_features_dict)
            atom_feature_list.append(features)
        edge_indices = []
        for bond in mol_no_h.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])
        heavy_atom_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
        pos_no_h = np.array(positions, dtype=np.float32)[heavy_atom_indices]
        return {'x': atom_feature_list, 'pos': pos_no_h.tolist(), 'edge_index': [list(e) for e in np.array(edge_indices, dtype=np.int64).T], 'num_nodes': mol_no_h.GetNumAtoms()}
    except Exception:
        return None
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess ZINC molecules into 2D graphs and 3D conformers', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\nExamples:\n  python zinc_process.py --zinc_csv data.csv --output_dir ./output\n  python zinc_process.py --zinc_csv data.csv --output_dir ./output --max_samples 1000\n  python zinc_process.py --zinc_csv data.csv --output_dir ./output --num_workers 8\n  python zinc_process.py --zinc_csv data.csv --output_dir ./output --timeout 1.0\n        ')
    parser.add_argument('--zinc_csv', type=str, required=True, help='Path to a ZINC CSV file with a smiles column')
    parser.add_argument('--output_dir', type=str, default='./data/zinc15', help='Output directory')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of molecules to process')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of worker processes')
    parser.add_argument('--timeout', type=float, default=0.5, help='Timeout per molecule in seconds')
    args = parser.parse_args()
    print('\n' + '=' * 80)
    print('ZINC preprocessing')
    print('=' * 80)
    print('\nNotes:')
    print('  Generates 2D molecular graphs and 3D conformers.')
    print('  Molecules that fail RDKit parsing or conformer generation are skipped.')
    print('  Output files are intended for Distill-Mol pretraining.')
    preprocessor = ZINCMultiModalPreprocessor(zinc_csv_path=args.zinc_csv, output_dir=args.output_dir, timeout=args.timeout)
    graphs_2d, conformers_3d, metadata = preprocessor.process_zinc_dataset(max_samples=args.max_samples, num_workers=args.num_workers)
    print(f"\n{'=' * 80}")
    print('Preprocessing complete')
    print(f"{'=' * 80}")
    print('\nProcessed objects:')
    print(f'   2D: {len(graphs_2d):,}')
    print(f'   3D: {len(conformers_3d):,}')
    print(f'   Metadata: {len(metadata):,}')
    print(f'\nOutput directory: {args.output_dir}')
    print(f'   2d_graphs.pkl')
    print(f'   3d_conformers.pkl')
    print(f'   metadata.pkl')
    print(f'   statistics.json')
    print(f'   processing_report.txt')
    print('\nOptional next step:')
    print(f'  Inspect the generated files in {args.output_dir}')
    print('')
