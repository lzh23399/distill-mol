"""Distill-Mol module."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict

class GraphPooling(nn.Module):
    """GraphPooling implementation."""

    def __init__(self, dim: int, pooling_type: str='attention'):
        super().__init__()
        self.pooling_type = pooling_type
        if pooling_type == 'attention':
            self.attention = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
        else:
            mask_expanded = torch.ones(h.shape[0], h.shape[1], 1, device=h.device)
        if self.pooling_type == 'mean':
            h_masked = h * mask_expanded
            return h_masked.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-08)
        elif self.pooling_type == 'attention':
            attn_scores = self.attention(h)
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=1)
            return (h * attn_weights).sum(dim=1)
        else:
            raise ValueError(f'Unknown pooling type: {self.pooling_type}')

class EGNNLayer(nn.Module):
    """EGNNLayer implementation."""

    def __init__(self, dim, edge_dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.edge_dim = edge_dim
        self.n_heads = n_heads
        self.edge_mlp = nn.Sequential(nn.Linear(dim * 2 + 1, edge_dim), nn.ReLU(), nn.Linear(edge_dim, edge_dim), nn.ReLU())
        self.message_mlp = nn.Sequential(nn.Linear(edge_dim + dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.coord_mlp = nn.Sequential(nn.Linear(edge_dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 1))
        self.node_mlp = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, h, x, mask=None):
        batch_size, n_nodes, _ = h.shape
        h_in = h
        rel_pos = x.unsqueeze(2) - x.unsqueeze(1)
        dist = torch.norm(rel_pos, dim=-1, keepdim=True) + 1e-08
        h_i = h.unsqueeze(2).expand(-1, -1, n_nodes, -1)
        h_j = h.unsqueeze(1).expand(-1, n_nodes, -1, -1)
        edge_input = torch.cat([h_i, h_j, dist], dim=-1)
        edge_feat = self.edge_mlp(edge_input)
        if mask is not None:
            edge_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
            edge_feat = edge_feat * edge_mask.unsqueeze(-1)
        coord_weights = self.coord_mlp(edge_feat)
        coord_diff = coord_weights * rel_pos / dist
        if mask is not None:
            coord_diff = coord_diff * edge_mask.unsqueeze(-1)
        x = x + coord_diff.sum(dim=2)
        messages = self.message_mlp(torch.cat([edge_feat, h_j], dim=-1))
        if mask is not None:
            messages = messages * edge_mask.unsqueeze(-1)
        h_msg = messages.sum(dim=2)
        h = self.node_mlp(torch.cat([h, h_msg], dim=-1))
        h = self.norm1(h + h_in)
        if mask is not None:
            h = h * mask.unsqueeze(-1)
            x = x * mask.unsqueeze(-1)
        attn = torch.softmax(coord_weights.squeeze(-1), dim=-1)
        return (h, x, attn)

class SE3Transformer(nn.Module):
    """SE3Transformer implementation."""

    def __init__(self, n_atom_types, dim=128, output_dim=256, edge_dim=64, n_layers=4, n_heads=8, dropout=0.1, k=16, pooling_type='attention'):
        super().__init__()
        self.n_atom_types = n_atom_types
        self.dim = dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.k = k
        self.atom_embedding = nn.Embedding(n_atom_types, dim)
        self.layers = nn.ModuleList([EGNNLayer(dim=dim, edge_dim=edge_dim, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)])
        self.pos_predictor = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, 3))
        self.pooling = GraphPooling(dim, pooling_type=pooling_type)
        self.output_projection = nn.Sequential(nn.Linear(dim, output_dim), nn.LayerNorm(output_dim), nn.ReLU(), nn.Dropout(dropout))
        self.aux_head = nn.Sequential(nn.Linear(output_dim, output_dim // 2), nn.ReLU(), nn.Linear(output_dim // 2, 3))
        print(f'3D teacher model: dim={dim}, output_dim={output_dim}, pooling={pooling_type}')

    def _get_graph_embedding(self, h: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """_get_graph_embedding helper."""
        graph_emb = self.pooling(h, mask)
        graph_emb = self.output_projection(graph_emb)
        return graph_emb

    def forward(self, atom_types, positions, mask=None, return_fixed_dim=False):
        """forward helper."""
        batch_size, n_atoms = atom_types.shape
        h = self.atom_embedding(atom_types)
        layer_features = []
        layer_attentions = []
        for layer in self.layers:
            h, positions, attn = layer(h, positions, mask)
            layer_features.append(h.clone())
            if attn is not None:
                layer_attentions.append(attn.clone())
        if return_fixed_dim:
            return self._get_graph_embedding(h, mask)
        pos_offset = self.pos_predictor(h)
        pred_positions = positions + pos_offset
        if mask is not None:
            pred_positions = pred_positions * mask.unsqueeze(-1)
        batch_features = []
        batch_attentions = []
        for b in range(batch_size):
            batch_features.append([f[b] for f in layer_features])
            batch_attentions.append([a[b] if a is not None else None for a in layer_attentions])
        return (pred_positions, (batch_features, batch_attentions))

    def forward_with_aux_loss(self, atom_types, positions, clean_positions, mask=None) -> Dict[str, torch.Tensor]:
        """forward_with_aux_loss helper."""
        batch_size, n_atoms = atom_types.shape
        h = self.atom_embedding(atom_types)
        layer_features = []
        layer_attentions = []
        for layer in self.layers:
            h, positions, attn = layer(h, positions, mask)
            layer_features.append(h.clone())
            if attn is not None:
                layer_attentions.append(attn.clone())
        pos_offset = self.pos_predictor(h)
        pred_positions = positions + pos_offset
        if mask is not None:
            pred_positions = pred_positions * mask.unsqueeze(-1)
            clean_positions_masked = clean_positions * mask.unsqueeze(-1)
        else:
            clean_positions_masked = clean_positions
        denoise_loss = F.mse_loss(pred_positions, clean_positions_masked)
        graph_embedding = self._get_graph_embedding(h, mask)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            centroid = (clean_positions * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-08)
        else:
            centroid = clean_positions.mean(dim=1)
        pred_centroid = self.aux_head(graph_embedding)
        aux_loss = F.mse_loss(pred_centroid, centroid)
        batch_features = []
        batch_attentions = []
        for b in range(batch_size):
            batch_features.append([f[b] for f in layer_features])
            batch_attentions.append([a[b] if a is not None else None for a in layer_attentions])
        return {'pred_positions': pred_positions, 'denoise_loss': denoise_loss, 'aux_loss': aux_loss, 'graph_embedding': graph_embedding, 'layer_features': batch_features, 'layer_attentions': batch_attentions}

    def forward_no_distill(self, atom_types, positions, mask=None):
        """forward_no_distill helper."""
        pred_positions, _ = self.forward(atom_types, positions, mask, return_fixed_dim=False)
        return pred_positions

    def get_fixed_dim_embedding(self, atom_types, positions, mask=None):
        """get_fixed_dim_embedding helper."""
        return self.forward(atom_types, positions, mask, return_fixed_dim=True)
if __name__ == '__main__':
    print('=' * 60)
    print('SE3Transformer')
    print('=' * 60)
    batch_size = 4
    n_atoms = 20
    n_atom_types = 10
    model = SE3Transformer(n_atom_types=n_atom_types, dim=128, output_dim=256, edge_dim=64, n_layers=4, pooling_type='attention')
    print(f'Model parameters: {sum((p.numel() for p in model.parameters())):,}')
    atom_types = torch.randint(0, n_atom_types, (batch_size, n_atoms))
    clean_positions = torch.randn(batch_size, n_atoms, 3)
    noisy_positions = clean_positions + 0.1 * torch.randn_like(clean_positions)
    mask = torch.ones(batch_size, n_atoms)
    print('\n[1] Forward pass')
    pred_pos, (feats, attns) = model(atom_types, noisy_positions, mask, return_fixed_dim=False)
    print(f'  Predicted position shape: {pred_pos.shape}')
    print('\n[2] Fixed-dimensional embedding')
    graph_emb = model.get_fixed_dim_embedding(atom_types, clean_positions, mask)
    print(f'  Graph embedding shape: {graph_emb.shape}')
    print('\n[3] Auxiliary losses')
    output = model.forward_with_aux_loss(atom_types, noisy_positions, clean_positions, mask)
    print(f"  Denoise loss: {output['denoise_loss'].item():.4f}")
    print(f"  Auxiliary loss: {output['aux_loss'].item():.4f}")
    print('\n[4] Check output projection gradients')
    model.zero_grad()
    output = model.forward_with_aux_loss(atom_types, noisy_positions, clean_positions, mask)
    total_loss = output['denoise_loss'] + 0.1 * output['aux_loss']
    total_loss.backward()
    proj_grad = model.output_projection[0].weight.grad
    if proj_grad is not None and proj_grad.abs().sum() > 0:
        print('   Output projection receives gradients.')
    else:
        print('   Warning: output projection did not receive gradients.')
    print('\nTeacher model smoke test complete.')
