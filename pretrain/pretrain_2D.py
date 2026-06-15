"""Distill-Mol module."""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import argparse
import json

class Graph2DDataset(Dataset):
    """Graph2DDataset implementation."""

    def __init__(self, graphs_2d_path, metadata_path):
        with open(graphs_2d_path, 'rb') as f:
            self.graphs = pickle.load(f)
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print(f'Loaded {len(self.graphs)} 2D graphs')

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph_data = self.graphs[idx]
        data = Data(x=graph_data['x'], edge_index=graph_data['edge_index'], edge_attr=graph_data['edge_attr'] if 'edge_attr' in graph_data else None)
        return data

class AttentionPooling(nn.Module):
    """AttentionPooling implementation."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x, batch):
        """forward helper."""
        attn_scores = self.attention(x)
        attn_weights = torch.zeros_like(attn_scores)
        for i in range(batch.max().item() + 1):
            mask = batch == i
            attn_weights[mask] = F.softmax(attn_scores[mask], dim=0)
        weighted_x = x * attn_weights
        graph_emb = global_add_pool(weighted_x, batch)
        return graph_emb

class GATEncoder(nn.Module):
    """GATEncoder implementation."""

    def __init__(self, node_feature_dim, hidden_dim=256, num_layers=4, num_heads=4, dropout=0.1, pooling_type='attention'):
        super(GATEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_dim = hidden_dim
            else:
                in_dim = hidden_dim * num_heads
            self.gat_layers.append(GATConv(in_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True if i < num_layers - 1 else False))
            out_dim = hidden_dim * num_heads if i < num_layers - 1 else hidden_dim
            self.batch_norms.append(nn.BatchNorm1d(out_dim))
        self.dropout = nn.Dropout(dropout)
        self.pooling_type = pooling_type
        if pooling_type == 'attention':
            self.pooling = AttentionPooling(hidden_dim)
        self.last_attention_weights = []

    def forward(self, x, edge_index, batch=None, return_attention_weights=False):
        """forward helper."""
        h = self.input_proj(x)
        h = F.relu(h)
        self.last_attention_weights = []
        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            if return_attention_weights:
                h, (edge_idx, alpha) = gat(h, edge_index, return_attention_weights=True)
                self.last_attention_weights.append({'layer': i, 'edge_index': edge_idx.detach(), 'attention': alpha.detach()})
            else:
                h = gat(h, edge_index)
            h = bn(h)
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = self.dropout(h)
        node_embeddings = h
        if batch is not None:
            if self.pooling_type == 'attention':
                graph_embedding = self.pooling(h, batch)
            else:
                graph_embedding = global_mean_pool(h, batch)
        else:
            graph_embedding = h.mean(dim=0, keepdim=True)
        if return_attention_weights:
            return (node_embeddings, graph_embedding, self.last_attention_weights)
        else:
            return (node_embeddings, graph_embedding)

class GraphMaskDecoder(nn.Module):
    """GraphMaskDecoder implementation."""

    def __init__(self, hidden_dim, node_feature_dim):
        super(GraphMaskDecoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, node_feature_dim))

    def forward(self, node_embeddings):
        return self.decoder(node_embeddings)

class GraphMaskedAutoencoder(nn.Module):
    """GraphMaskedAutoencoder implementation."""

    def __init__(self, node_feature_dim, hidden_dim=256, output_dim=256, num_layers=4, num_heads=4, dropout=0.1, mask_ratio=0.15, pooling_type='attention'):
        super(GraphMaskedAutoencoder, self).__init__()
        self.mask_ratio = mask_ratio
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.encoder = GATEncoder(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout, pooling_type=pooling_type)
        self.decoder = GraphMaskDecoder(hidden_dim, node_feature_dim)
        self.mask_token = nn.Parameter(torch.randn(node_feature_dim))
        self.output_projection = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim), nn.ReLU(), nn.Dropout(dropout))
        self.aux_head = nn.Sequential(nn.Linear(output_dim, output_dim // 2), nn.ReLU(), nn.Linear(output_dim // 2, 3))
        print(f' 2D: hidden_dim={hidden_dim}, output_dim={output_dim}, pooling={pooling_type}')

    def mask_nodes(self, x, num_nodes_per_graph):
        """mask_nodes helper."""
        device = x.device
        total_nodes = x.shape[0]
        mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)
        start_idx = 0
        for num_nodes in num_nodes_per_graph:
            end_idx = start_idx + num_nodes
            num_mask = max(1, int(num_nodes * self.mask_ratio))
            perm = torch.randperm(num_nodes, device=device)
            mask_indices = perm[:num_mask] + start_idx
            mask[mask_indices] = True
            start_idx = end_idx
        masked_x = x.clone()
        masked_x[mask] = self.mask_token.unsqueeze(0).expand(mask.sum(), -1)
        return (masked_x, mask)

    def _compute_graph_stats(self, data):
        """_compute_graph_stats helper."""
        batch = data.batch
        batch_size = batch.max().item() + 1
        stats = torch.zeros(batch_size, 3, device=batch.device)
        for i in range(batch_size):
            node_mask = batch == i
            num_nodes = node_mask.sum().float()
            stats[i, 0] = num_nodes / 50.0
            if hasattr(data, 'x') and data.x is not None:
                node_feat_mean = data.x[node_mask].mean()
                stats[i, 1] = node_feat_mean
            if hasattr(data, 'x') and data.x is not None:
                node_feat_std = data.x[node_mask].std()
                stats[i, 2] = node_feat_std
        return stats

    def forward(self, data, return_embeddings=False, return_attention_weights=False):
        """forward helper."""
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        if return_embeddings:
            if return_attention_weights:
                node_embeddings, graph_embeddings, attentions = self.encoder(x, edge_index, batch, return_attention_weights=True)
                graph_embeddings_fixed = self.output_projection(graph_embeddings)
                return (node_embeddings, graph_embeddings_fixed, attentions)
            else:
                node_embeddings, graph_embeddings = self.encoder(x, edge_index, batch)
                graph_embeddings_fixed = self.output_projection(graph_embeddings)
                return (node_embeddings, graph_embeddings_fixed)
        num_nodes_per_graph = torch.bincount(batch)
        masked_x, mask = self.mask_nodes(x, num_nodes_per_graph)
        if return_attention_weights:
            node_embeddings, graph_embeddings, attentions = self.encoder(masked_x, edge_index, batch, return_attention_weights=True)
            reconstructed = self.decoder(node_embeddings[mask])
            target = x[mask]
            loss = F.mse_loss(reconstructed, target)
            return (loss, node_embeddings, graph_embeddings, attentions)
        else:
            node_embeddings, graph_embeddings = self.encoder(masked_x, edge_index, batch)
            reconstructed = self.decoder(node_embeddings[mask])
            target = x[mask]
            loss = F.mse_loss(reconstructed, target)
            return (loss, node_embeddings, graph_embeddings)

    def forward_with_aux_loss(self, data):
        """forward_with_aux_loss helper."""
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        num_nodes_per_graph = torch.bincount(batch)
        masked_x, mask = self.mask_nodes(x, num_nodes_per_graph)
        node_embeddings, graph_embeddings = self.encoder(masked_x, edge_index, batch)
        reconstructed = self.decoder(node_embeddings[mask])
        target = x[mask]
        reconstruction_loss = F.mse_loss(reconstructed, target)
        graph_embeddings_fixed = self.output_projection(graph_embeddings)
        graph_stats = self._compute_graph_stats(data)
        pred_stats = self.aux_head(graph_embeddings_fixed)
        aux_loss = F.mse_loss(pred_stats, graph_stats)
        return {'reconstruction_loss': reconstruction_loss, 'aux_loss': aux_loss, 'total_loss': reconstruction_loss + 0.1 * aux_loss, 'graph_embedding': graph_embeddings_fixed, 'node_embeddings': node_embeddings}

    def encode_graph(self, x, edge_index, batch):
        """encode_graph helper."""
        node_embeddings, graph_embeddings = self.encoder(x, edge_index, batch)
        graph_embeddings_fixed = self.output_projection(graph_embeddings)
        return (node_embeddings, graph_embeddings_fixed)

    def get_attention_weights(self):
        """get_attention_weights helper."""
        return self.encoder.last_attention_weights

def train_epoch(model, dataloader, optimizer, device, use_aux_loss=True):
    """train_epoch helper."""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_aux_loss = 0
    num_batches = 0
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        batch = batch.to(device)
        optimizer.zero_grad()
        if use_aux_loss and hasattr(model, 'forward_with_aux_loss'):
            output = model.forward_with_aux_loss(batch)
            loss = output['total_loss']
            recon_loss = output['reconstruction_loss'].item()
            aux_loss = output['aux_loss'].item()
            total_recon_loss += recon_loss
            total_aux_loss += aux_loss
        else:
            loss, _, _ = model(batch, return_embeddings=False)
            recon_loss = loss.item()
            aux_loss = 0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'recon': f'{recon_loss:.4f}', 'aux': f'{aux_loss:.4f}'})
    return {'total_loss': total_loss / num_batches, 'recon_loss': total_recon_loss / num_batches, 'aux_loss': total_aux_loss / num_batches}

def validate(model, dataloader, device, use_aux_loss=True):
    """validate helper."""
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            batch = batch.to(device)
            if use_aux_loss and hasattr(model, 'forward_with_aux_loss'):
                output = model.forward_with_aux_loss(batch)
                loss = output['total_loss'].item()
            else:
                loss, _, _ = model(batch, return_embeddings=False)
                loss = loss.item()
            total_loss += loss
            num_batches += 1
    return total_loss / num_batches

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    checkpoint_dir = os.path.join(args.checkpoint_dir, '2d_gat')
    os.makedirs(checkpoint_dir, exist_ok=True)
    print('\nLoading 2D graph data...')
    dataset = Graph2DDataset(graphs_2d_path=os.path.join(args.data_dir, '2d_graphs.pkl'), metadata_path=os.path.join(args.data_dir, 'metadata.pkl'))
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: Batch.from_data_list(x))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: Batch.from_data_list(x))
    print('\nCreating model...')
    sample_data = dataset[0]
    node_feature_dim = sample_data.x.shape[1]
    model = GraphMaskedAutoencoder(node_feature_dim=node_feature_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, mask_ratio=args.mask_ratio, pooling_type='attention').to(device)
    print(f'Model parameters: {sum((p.numel() for p in model.parameters())):,}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    print('\nStarting training...')
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'train_recon_loss': [], 'train_aux_loss': [], 'val_loss': []}
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        train_metrics = train_epoch(model, train_loader, optimizer, device, use_aux_loss=True)
        print(f"Train - Total: {train_metrics['total_loss']:.4f}, Recon: {train_metrics['recon_loss']:.4f}, Aux: {train_metrics['aux_loss']:.4f}")
        val_loss = validate(model, val_loader, device, use_aux_loss=True)
        print(f'Val Loss: {val_loss:.4f}')
        history['train_loss'].append(train_metrics['total_loss'])
        history['train_recon_loss'].append(train_metrics['recon_loss'])
        history['train_aux_loss'].append(train_metrics['aux_loss'])
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_loss': val_loss, 'hidden_dim': args.hidden_dim, 'output_dim': args.output_dim, 'node_feature_dim': node_feature_dim}, checkpoint_path)
            print(f'Saved best model to {checkpoint_path}')
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs')
            break
    print('\nTraining complete!')
    print(f'Best validation loss: {best_val_loss:.4f}')
    config = {'model_type': '2d_gat_improved', 'node_feature_dim': node_feature_dim, 'hidden_dim': args.hidden_dim, 'output_dim': args.output_dim, 'num_layers': args.num_layers, 'num_heads': args.num_heads, 'dropout': args.dropout, 'mask_ratio': args.mask_ratio, 'best_val_loss': best_val_loss}
    config_path = os.path.join(checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    history_path = os.path.join(checkpoint_dir, 'train_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'Saved config to {config_path}')
    print('\nChecking output projection gradients...')
    model.zero_grad()
    sample_batch = next(iter(train_loader)).to(device)
    output = model.forward_with_aux_loss(sample_batch)
    output['total_loss'].backward()
    proj_grad = model.output_projection[0].weight.grad
    if proj_grad is not None and proj_grad.abs().sum() > 0:
        print('Output projection receives gradients.')
    else:
        print('Warning: output projection did not receive gradients.')
    print('')
    model.eval()
    test_batch = next(iter(val_loader)).to(device)
    with torch.no_grad():
        _, _, attentions = model(test_batch, return_embeddings=True, return_attention_weights=True)
    if len(attentions) > 0:
        print(f'Collected attention weights from {len(attentions)} layers.')
        for i, attn_info in enumerate(attentions):
            print(f"  - Layer {i}: edge_index.shape={attn_info['edge_index'].shape}, attention.shape={attn_info['attention'].shape}")
    else:
        print('No attention weights were returned.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D Graph Masked Pretraining (Improved)')
    parser.add_argument('--data_dir', type=str, default='../data/zinc15')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--output_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-05)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()
    main(args)
