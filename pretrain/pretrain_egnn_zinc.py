"""Distill-Mol module."""
import os
import sys
import pickle
import argparse
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class ZINCConformer3DDataset(Dataset):
    """ZINCConformer3DDataset implementation."""

    def __init__(self, conformers_path, metadata_path=None, max_atom_type=100):
        with open(conformers_path, 'rb') as f:
            self.conformers = pickle.load(f)
        self.metadata = None
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        print(f'Loaded {len(self.conformers)} conformers')
        self.atom_type_mapping = self._build_atom_type_mapping(max_atom_type)
        self.n_atom_types = len(self.atom_type_mapping)
        print(f'Atom type mapping preview: {dict(list(self.atom_type_mapping.items())[:10])}')
        print(f'Number of atom types: {self.n_atom_types}')

    def _build_atom_type_mapping(self, max_atom_type):
        """_build_atom_type_mapping helper."""
        all_atom_types = set()
        total_conformers = len(self.conformers)
        sample_size = min(total_conformers, 50000)
        print(f'Scanning {sample_size}/{total_conformers} conformers for atom types...')
        if total_conformers > sample_size:
            indices = np.linspace(0, total_conformers - 1, sample_size, dtype=int)
            print('Using an evenly spaced conformer sample.')
        else:
            indices = range(total_conformers)
            print('Using all conformers.')
        for i, idx in enumerate(indices):
            atom_types = self._extract_atom_types(self.conformers[idx])
            if atom_types is not None:
                all_atom_types.update(atom_types.tolist())
            if (i + 1) % 10000 == 0:
                print(f'     Scanned {i + 1}/{len(indices)} conformers; unique atom types={len(all_atom_types)}')
        sorted_types = sorted(list(all_atom_types))
        print(f'Found {len(sorted_types)} unique atom types')
        print(f'    ID: [{min(sorted_types)}, {max(sorted_types)}]')
        if len(sorted_types) > max_atom_type:
            print(f'Atom type count exceeds the limit: {len(sorted_types)} > {max_atom_type}')
            print(f'Keeping the first {max_atom_type} atom type IDs')
            print(f'Skipped atom type IDs: {sorted_types[max_atom_type:]}')
            sorted_types = sorted_types[:max_atom_type]
        atom_type_mapping = {orig_type: new_idx for new_idx, orig_type in enumerate(sorted_types)}
        print('First 10 atom type mappings:')
        for orig_type, new_idx in list(atom_type_mapping.items())[:10]:
            print(f'    original ID {orig_type:3d} -> mapped ID {new_idx:3d}')
        return atom_type_mapping

    def _extract_atom_types(self, conf):
        if 'atom_types' in conf:
            return conf['atom_types']
        if 'x' in conf:
            atom_feat = conf['x']
            if atom_feat.shape[1] >= 10:
                one_hot = atom_feat[:, :10]
                if torch.all((one_hot == 0) | (one_hot == 1)):
                    return torch.argmax(one_hot, dim=1)
                for n_types in [20, 30, 50, 100]:
                    if atom_feat.shape[1] >= n_types:
                        one_hot = atom_feat[:, :n_types]
                        if torch.all((one_hot >= 0) & (one_hot <= 1)):
                            return torch.argmax(one_hot, dim=1)
        if 'z' in conf:
            return conf['z']
        if 'pos' in conf:
            num_atoms = conf['pos'].shape[0]
            return torch.full((num_atoms,), 6, dtype=torch.long)
        return None

    def __len__(self):
        return len(self.conformers)

    def __getitem__(self, idx):
        conf = self.conformers[idx]
        pos = conf['pos']
        atom_types_raw = self._extract_atom_types(conf)
        if atom_types_raw is None:
            atom_types_raw = torch.full((pos.shape[0],), 6, dtype=torch.long)
        atom_types = torch.zeros_like(atom_types_raw)
        for i, orig_type in enumerate(atom_types_raw.tolist()):
            if orig_type in self.atom_type_mapping:
                atom_types[i] = self.atom_type_mapping[orig_type]
            else:
                atom_types[i] = 0
        return {'atom_types': atom_types, 'pos': pos, 'num_atoms': pos.shape[0]}

def collate_fn_padding(batch):
    """collate_fn_padding helper."""
    batch_size = len(batch)
    max_atoms = max([sample['num_atoms'] for sample in batch])
    atom_types = torch.zeros(batch_size, max_atoms, dtype=torch.long)
    positions = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_atoms, dtype=torch.float32)
    for i, sample in enumerate(batch):
        n_atoms = sample['num_atoms']
        atom_types[i, :n_atoms] = sample['atom_types']
        positions[i, :n_atoms] = sample['pos']
        mask[i, :n_atoms] = 1.0
    return {'atom_types': atom_types, 'positions': positions, 'mask': mask}

class ConformerNoiseGenerator:
    """ConformerNoiseGenerator implementation."""

    @staticmethod
    def rotation_matrix_3d(axis, angle):
        axis = axis / (torch.norm(axis) + 1e-08)
        K = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], device=axis.device, dtype=axis.dtype)
        I = torch.eye(3, device=axis.device, dtype=axis.dtype)
        R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        return R

    @staticmethod
    def rotation_noise(positions, max_angle_deg=30.0):
        device = positions.device
        dtype = positions.dtype
        is_batched = positions.dim() == 3
        if not is_batched:
            positions = positions.unsqueeze(0)
        batch_size = positions.shape[0]
        noisy_positions = []
        for i in range(batch_size):
            pos = positions[i]
            axis = torch.randn(3, device=device, dtype=dtype)
            angle = torch.rand(1, device=device, dtype=dtype) * max_angle_deg * np.pi / 180.0
            R = ConformerNoiseGenerator.rotation_matrix_3d(axis, angle)
            pos_rotated = pos @ R.T
            noisy_positions.append(pos_rotated)
        noisy_positions = torch.stack(noisy_positions, dim=0)
        if not is_batched:
            noisy_positions = noisy_positions.squeeze(0)
        return noisy_positions

    @staticmethod
    def vibration_rotation_noise(positions, vibration_scale=0.1, max_angle_deg=30.0):
        vibration = torch.randn_like(positions) * vibration_scale
        positions_vibrated = positions + vibration
        return ConformerNoiseGenerator.rotation_noise(positions_vibrated, max_angle_deg=max_angle_deg)

    @staticmethod
    def gaussian_noise(positions, noise_scale=0.2):
        noise = torch.randn_like(positions) * noise_scale
        return positions + noise

class EGNNDistillationTrainer:
    """EGNNDistillationTrainer implementation."""

    def __init__(self, n_atom_types, teacher_config=None, student_config=None, device='cuda'):
        self.device = device
        self.n_atom_types = n_atom_types
        if teacher_config is None:
            teacher_config = {'dim': 128, 'output_dim': 256, 'edge_dim': 64, 'n_layers': 4, 'n_heads': 8, 'dropout': 0.1, 'k': 16, 'pooling_type': 'attention'}
        if student_config is None:
            student_config = {'dim': 64, 'output_dim': 256, 'edge_dim': 32, 'n_layers': 2, 'n_heads': 4, 'dropout': 0.1, 'k': 12, 'pooling_type': 'attention'}
        self.teacher_config = teacher_config
        self.student_config = student_config
        self.teacher_model = self._create_teacher_model().to(device)
        self.student_model = self._create_student_model().to(device)
        self.noise_gen = ConformerNoiseGenerator()
        print('Model parameter counts:')
        print(f'  Teacher parameters: {sum((p.numel() for p in self.teacher_model.parameters())):,}')
        print(f'  Student parameters: {sum((p.numel() for p in self.student_model.parameters())):,}')

    def _create_teacher_model(self):
        try:
            sys.path.append('./egnn-pytorch-main')
            sys.path.append('./models')
            sys.path.append('../models')
            from teacher_model import SE3Transformer
            print('Created SE3Transformer teacher model')
            return SE3Transformer(n_atom_types=self.n_atom_types, **self.teacher_config)
        except ImportError as e:
            print(f'Failed to import SE3Transformer; using placeholder model. Error: {e}')
            return self._placeholder_model(self.teacher_config['dim'])

    def _create_student_model(self):
        try:
            from student_model import SE3TransformerStudent
            print('Created SE3TransformerStudent model')
            model = SE3TransformerStudent(n_atom_types=self.n_atom_types, **self.student_config)
            model.set_teacher_dim(self.teacher_config['dim'])
            return model
        except ImportError as e:
            print(f'Failed to import SE3TransformerStudent; using placeholder model. Error: {e}')
            return self._placeholder_model(self.student_config['dim'])

    def _placeholder_model(self, dim):

        class PlaceholderModel(nn.Module):

            def __init__(self, n_types, dim):
                super().__init__()
                self.embedding = nn.Embedding(n_types, dim)
                self.predictor = nn.Linear(dim, 3)

            def forward(self, atom_types, positions, mask=None):
                h = self.embedding(atom_types)
                pred_offset = self.predictor(h)
                pred_pos = positions + pred_offset
                if mask is not None:
                    pred_pos = pred_pos * mask.unsqueeze(-1)
                return (pred_pos, ([], []))

            def forward_with_aux_loss(self, atom_types, positions, clean_positions, mask=None):
                pred_pos, _ = self.forward(atom_types, positions, mask)
                loss = F.mse_loss(pred_pos, clean_positions)
                return {'pred_positions': pred_pos, 'denoise_loss': loss, 'aux_loss': torch.tensor(0.0), 'total_loss': loss}
        return PlaceholderModel(self.n_atom_types, dim)

    def _validate_batch(self, batch):
        atom_types = batch['atom_types']
        max_atom_type = atom_types.max().item()
        if max_atom_type >= self.n_atom_types:
            atom_types = torch.clamp(atom_types, 0, self.n_atom_types - 1)
            batch['atom_types'] = atom_types
        return batch

    def train_teacher(self, dataloader, num_epochs=50, learning_rate=0.0001, noise_type='vrn', aux_loss_weight=0.1, save_dir='./checkpoints/egnn_teacher'):
        """train_teacher helper."""
        os.makedirs(save_dir, exist_ok=True)
        self.teacher_model.train()
        optimizer = torch.optim.AdamW(self.teacher_model.parameters(), lr=learning_rate, weight_decay=1e-05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-06)
        print('\n' + '=' * 60)
        print(f'Training teacher with noise type: {noise_type.upper()}')
        print(f'Auxiliary loss weight: {aux_loss_weight}')
        print('=' * 60)
        best_loss = float('inf')
        history = {'train_loss': [], 'denoise_loss': [], 'aux_loss': []}
        for epoch in range(num_epochs):
            epoch_total_loss = 0.0
            epoch_denoise_loss = 0.0
            epoch_aux_loss = 0.0
            num_batches = 0
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')
            for batch_idx, batch in enumerate(progress_bar):
                batch = self._validate_batch(batch)
                atom_types = batch['atom_types'].to(self.device)
                pos_clean = batch['positions'].to(self.device)
                mask = batch['mask'].to(self.device)
                if noise_type == 'rn':
                    pos_noisy = self.noise_gen.rotation_noise(pos_clean)
                elif noise_type == 'vrn':
                    pos_noisy = self.noise_gen.vibration_rotation_noise(pos_clean)
                else:
                    pos_noisy = self.noise_gen.gaussian_noise(pos_clean)
                pos_noisy = pos_noisy * mask.unsqueeze(-1)
                pos_clean = pos_clean * mask.unsqueeze(-1)
                optimizer.zero_grad()
                try:
                    if hasattr(self.teacher_model, 'forward_with_aux_loss'):
                        output = self.teacher_model.forward_with_aux_loss(atom_types, pos_noisy, pos_clean, mask)
                        denoise_loss = output['denoise_loss']
                        aux_loss = output['aux_loss']
                        total_loss = denoise_loss + aux_loss_weight * aux_loss
                    else:
                        pred_pos, _ = self.teacher_model(atom_types, pos_noisy, mask)
                        denoise_loss = F.mse_loss(pred_pos * mask.unsqueeze(-1), pos_clean * mask.unsqueeze(-1))
                        aux_loss = torch.tensor(0.0, device=self.device)
                        total_loss = denoise_loss
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.teacher_model.parameters(), 1.0)
                    optimizer.step()
                    epoch_total_loss += total_loss.item()
                    epoch_denoise_loss += denoise_loss.item()
                    epoch_aux_loss += aux_loss.item()
                    num_batches += 1
                    progress_bar.set_postfix({'total': f'{total_loss.item():.4f}', 'denoise': f'{denoise_loss.item():.4f}', 'aux': f'{aux_loss.item():.4f}'})
                except RuntimeError as e:
                    print(f'\nRuntime error in teacher batch; skipping batch: {e}')
                    continue
            avg_total = epoch_total_loss / max(num_batches, 1)
            avg_denoise = epoch_denoise_loss / max(num_batches, 1)
            avg_aux = epoch_aux_loss / max(num_batches, 1)
            history['train_loss'].append(avg_total)
            history['denoise_loss'].append(avg_denoise)
            history['aux_loss'].append(avg_aux)
            scheduler.step()
            print(f'Epoch {epoch + 1}/{num_epochs} - total: {avg_total:.4f}, denoise: {avg_denoise:.4f}, aux: {avg_aux:.4f}')
            if avg_total < best_loss:
                best_loss = avg_total
                checkpoint = {'epoch': epoch, 'model_state_dict': self.teacher_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_total, 'config': self.teacher_config, 'n_atom_types': self.n_atom_types, 'output_dim': self.teacher_config.get('output_dim', 256)}
                torch.save(checkpoint, os.path.join(save_dir, 'teacher_best.pt'))
                print(f'    Saved best teacher checkpoint (loss: {avg_total:.4f})')
        torch.save(self.teacher_model.state_dict(), os.path.join(save_dir, 'teacher_final.pt'))
        with open(os.path.join(save_dir, 'train_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        self._plot_training_curves(history, save_dir, 'teacher')
        print(f'\nTeacher training complete. Best loss: {best_loss:.4f}')
        return (best_loss, history)

    def train_student_with_distillation(self, dataloader, teacher_checkpoint, num_epochs=100, learning_rate=0.0001, distill_alpha=0.5, temperature=2.0, aux_loss_weight=0.1, save_dir='./checkpoints/egnn_student'):
        """train_student_with_distillation helper."""
        os.makedirs(save_dir, exist_ok=True)
        print(f'\nLoading teacher checkpoint: {teacher_checkpoint}')
        checkpoint = torch.load(teacher_checkpoint, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.teacher_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.teacher_model.load_state_dict(checkpoint)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        print('Teacher model loaded and frozen.')
        self.student_model.train()
        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=learning_rate, weight_decay=1e-05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-06)
        print('\n' + '=' * 60)
        print('Training student with teacher distillation')
        print(f'Distillation alpha: {distill_alpha}, temperature: {temperature}')
        print('=' * 60)
        best_loss = float('inf')
        history = {'train_loss': [], 'denoise_loss': [], 'distill_loss': [], 'aux_loss': []}
        for epoch in range(num_epochs):
            epoch_total_loss = 0.0
            epoch_denoise_loss = 0.0
            epoch_distill_loss = 0.0
            epoch_aux_loss = 0.0
            num_batches = 0
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')
            for batch in progress_bar:
                batch = self._validate_batch(batch)
                atom_types = batch['atom_types'].to(self.device)
                pos_clean = batch['positions'].to(self.device)
                mask = batch['mask'].to(self.device)
                pos_noisy = self.noise_gen.gaussian_noise(pos_clean, noise_scale=0.2)
                pos_noisy = pos_noisy * mask.unsqueeze(-1)
                optimizer.zero_grad()
                try:
                    with torch.no_grad():
                        if hasattr(self.teacher_model, 'forward_with_aux_loss'):
                            teacher_output = self.teacher_model.forward_with_aux_loss(atom_types, pos_clean, pos_clean, mask)
                            teacher_feats = teacher_output.get('layer_features', None)
                        else:
                            _, (teacher_feats, _) = self.teacher_model(atom_types, pos_clean, mask)
                    if hasattr(self.student_model, 'forward_with_aux_loss'):
                        student_output = self.student_model.forward_with_aux_loss(atom_types, pos_noisy, pos_clean, mask, teacher_features=teacher_feats, distill_weight=distill_alpha, temperature=temperature)
                        denoise_loss = student_output['denoise_loss']
                        distill_loss = student_output.get('distill_loss', torch.tensor(0.0))
                        aux_loss = student_output.get('aux_loss', torch.tensor(0.0))
                        total_loss = student_output['total_loss']
                    else:
                        pred_pos, (student_feats, _) = self.student_model(atom_types, pos_noisy, mask)
                        denoise_loss = F.mse_loss(pred_pos * mask.unsqueeze(-1), pos_clean * mask.unsqueeze(-1))
                        distill_loss = torch.tensor(0.0, device=self.device)
                        aux_loss = torch.tensor(0.0, device=self.device)
                        total_loss = denoise_loss
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                    optimizer.step()
                    epoch_total_loss += total_loss.item()
                    epoch_denoise_loss += denoise_loss.item()
                    epoch_distill_loss += distill_loss.item() if torch.is_tensor(distill_loss) else distill_loss
                    epoch_aux_loss += aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss
                    num_batches += 1
                    progress_bar.set_postfix({'total': f'{total_loss.item():.4f}', 'denoise': f'{denoise_loss.item():.4f}'})
                except RuntimeError as e:
                    print(f'\nRuntime error in student batch; skipping batch: {e}')
                    continue
            avg_total = epoch_total_loss / max(num_batches, 1)
            avg_denoise = epoch_denoise_loss / max(num_batches, 1)
            avg_distill = epoch_distill_loss / max(num_batches, 1)
            avg_aux = epoch_aux_loss / max(num_batches, 1)
            history['train_loss'].append(avg_total)
            history['denoise_loss'].append(avg_denoise)
            history['distill_loss'].append(avg_distill)
            history['aux_loss'].append(avg_aux)
            scheduler.step()
            print(f'Epoch {epoch + 1}/{num_epochs} - total: {avg_total:.4f}, denoise: {avg_denoise:.4f}, distill: {avg_distill:.4f}')
            if avg_total < best_loss:
                best_loss = avg_total
                checkpoint = {'epoch': epoch, 'model_state_dict': self.student_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_total, 'config': self.student_config, 'n_atom_types': self.n_atom_types, 'output_dim': self.student_config.get('output_dim', 256)}
                torch.save(checkpoint, os.path.join(save_dir, 'student_best.pt'))
                print(f'    Saved best student checkpoint (loss: {avg_total:.4f})')
        torch.save(self.student_model.state_dict(), os.path.join(save_dir, 'student_final.pt'))
        with open(os.path.join(save_dir, 'train_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        self._plot_training_curves(history, save_dir, 'student')
        print(f'\nStudent training complete. Best loss: {best_loss:.4f}')
        return (best_loss, history)

    def _plot_training_curves(self, history, save_dir, model_type='teacher'):
        """_plot_training_curves helper."""
        n_plots = len([k for k in history.keys() if history[k]])
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        plot_idx = 0
        for key, values in history.items():
            if values:
                axes[plot_idx].plot(values)
                axes[plot_idx].set_xlabel('Epoch')
                axes[plot_idx].set_ylabel(key)
                axes[plot_idx].set_title(f'{model_type} {key}')
                axes[plot_idx].grid(True)
                plot_idx += 1
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_type}_training_curves.png'), dpi=150)
        plt.close()
        print(f'Saved training curves to {save_dir}')

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print('\nLoading ZINC 3D conformer dataset...')
    dataset = ZINCConformer3DDataset(conformers_path=os.path.join(args.data_dir, '3d_conformers.pkl'), metadata_path=os.path.join(args.data_dir, 'metadata.pkl'), max_atom_type=args.max_atom_types)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn_padding, pin_memory=True if device.type == 'cuda' else False)
    n_atom_types = dataset.n_atom_types
    teacher_config = {'dim': args.teacher_dim, 'output_dim': args.output_dim, 'edge_dim': args.teacher_edge_dim, 'n_layers': args.teacher_layers, 'n_heads': args.teacher_heads, 'dropout': args.dropout, 'k': args.teacher_k, 'pooling_type': 'attention'}
    student_config = {'dim': args.student_dim, 'output_dim': args.output_dim, 'edge_dim': args.student_edge_dim, 'n_layers': args.student_layers, 'n_heads': args.student_heads, 'dropout': args.dropout, 'k': args.student_k, 'pooling_type': 'attention'}
    trainer = EGNNDistillationTrainer(n_atom_types=n_atom_types, teacher_config=teacher_config, student_config=student_config, device=device)
    if args.train_teacher:
        print('\n' + '=' * 60)
        print('Stage 1: teacher pretraining')
        print('=' * 60)
        trainer.train_teacher(dataloader=dataloader, num_epochs=args.teacher_epochs, learning_rate=args.learning_rate, noise_type='vrn', aux_loss_weight=args.aux_loss_weight, save_dir=args.teacher_save_dir)
    if args.train_student:
        print('\n' + '=' * 60)
        print('Stage 2: student distillation')
        print('=' * 60)
        teacher_checkpoint = os.path.join(args.teacher_save_dir, 'teacher_best.pt')
        if not os.path.exists(teacher_checkpoint):
            teacher_checkpoint = os.path.join(args.teacher_save_dir, 'teacher_final.pt')
        if not os.path.exists(teacher_checkpoint):
            print(f'Teacher checkpoint not found: {teacher_checkpoint}')
            return
        trainer.train_student_with_distillation(dataloader=dataloader, teacher_checkpoint=teacher_checkpoint, num_epochs=args.student_epochs, learning_rate=args.learning_rate, distill_alpha=args.distill_alpha, temperature=args.temperature, aux_loss_weight=args.aux_loss_weight, save_dir=args.student_save_dir)
    print('\nEGNN pretraining complete.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EGNN teacher-student pretraining on ZINC conformers')
    parser.add_argument('--data_dir', type=str, default='../data/zinc15/all')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_atom_types', type=int, default=100)
    parser.add_argument('--output_dim', type=int, default=256, help='Output embedding dimension')
    parser.add_argument('--teacher_dim', type=int, default=128)
    parser.add_argument('--teacher_edge_dim', type=int, default=64)
    parser.add_argument('--teacher_layers', type=int, default=4)
    parser.add_argument('--teacher_heads', type=int, default=8)
    parser.add_argument('--teacher_k', type=int, default=16)
    parser.add_argument('--teacher_epochs', type=int, default=50)
    parser.add_argument('--teacher_save_dir', type=str, default='./check_all/egnn_teacher')
    parser.add_argument('--student_dim', type=int, default=64)
    parser.add_argument('--student_edge_dim', type=int, default=32)
    parser.add_argument('--student_layers', type=int, default=2)
    parser.add_argument('--student_heads', type=int, default=4)
    parser.add_argument('--student_k', type=int, default=12)
    parser.add_argument('--student_epochs', type=int, default=100)
    parser.add_argument('--student_save_dir', type=str, default='./check_all/egnn_student')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--distill_alpha', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--aux_loss_weight', type=float, default=0.1, help='Auxiliary centroid prediction loss weight')
    parser.add_argument('--train_teacher', action='store_true', default=False)
    parser.add_argument('--train_student', action='store_true', default=False)
    parser.add_argument('--train_both', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.train_both:
        args.train_teacher = True
        args.train_student = True
    if not args.train_teacher and (not args.train_student):
        print('No training stage was selected; defaulting to teacher pretraining.')
        args.train_teacher = True
    main(args)
