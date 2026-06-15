"""Distill-Mol module."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List

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

class EGNNLayerStudent(nn.Module):
    """EGNNLayerStudent implementation."""

    def __init__(self, dim, edge_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.edge_dim = edge_dim
        self.edge_mlp = nn.Sequential(nn.Linear(dim * 2 + 1, edge_dim), nn.ReLU(), nn.Linear(edge_dim, edge_dim))
        self.message_mlp = nn.Sequential(nn.Linear(edge_dim + dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.coord_mlp = nn.Sequential(nn.Linear(edge_dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 1))
        self.node_mlp = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

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
        h = self.norm(h + h_in)
        if mask is not None:
            h = h * mask.unsqueeze(-1)
            x = x * mask.unsqueeze(-1)
        attn = torch.softmax(coord_weights.squeeze(-1), dim=-1)
        return (h, x, attn)

class SE3TransformerStudent(nn.Module):
    """SE3TransformerStudent implementation."""

    def __init__(self, n_atom_types, dim=64, output_dim=None, edge_dim=32, n_layers=2, n_heads=4, dropout=0.1, k=12, pooling_type='attention'):
        super().__init__()
        self.n_atom_types = n_atom_types
        self.dim = dim
        self.output_dim = output_dim if output_dim is not None else dim
        self.n_layers = n_layers
        self.k = k
        self.atom_embedding = nn.Embedding(n_atom_types, dim)
        self.layers = nn.ModuleList([EGNNLayerStudent(dim=dim, edge_dim=edge_dim, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)])
        self.pos_predictor = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 3))
        self.pooling = GraphPooling(dim, pooling_type=pooling_type)
        if output_dim is not None and output_dim != dim:
            self.output_projection = nn.Sequential(nn.Linear(dim, output_dim), nn.LayerNorm(output_dim), nn.ReLU(), nn.Dropout(dropout))
            self.aux_head = nn.Sequential(nn.Linear(output_dim, output_dim // 2), nn.ReLU(), nn.Linear(output_dim // 2, 3))
            print(f'3D student model: dim={dim}, output_dim={output_dim}, pooling={pooling_type}')
        else:
            self.output_projection = None
            self.aux_head = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 3))
            print(f'3D student model: dim={dim}, output_dim={self.output_dim}, pooling={pooling_type}')
        self.feature_aligner = None

    def set_teacher_dim(self, teacher_dim: int):
        """set_teacher_dim helper."""
        if self.dim != teacher_dim:
            self.feature_aligner = nn.Linear(self.dim, teacher_dim)
            print(f'Feature aligner: {self.dim} -> {teacher_dim}')

    def _get_graph_embedding(self, h: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """_get_graph_embedding helper."""
        graph_emb = self.pooling(h, mask)
        if self.output_projection is not None:
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

    def forward_with_aux_loss(self, atom_types, positions, clean_positions, mask=None, teacher_features=None, distill_weight=0.5, temperature=2.0) -> Dict[str, torch.Tensor]:
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
        result = {'pred_positions': pred_positions, 'denoise_loss': denoise_loss, 'aux_loss': aux_loss, 'graph_embedding': graph_embedding}
        if teacher_features is not None:
            distill_loss = self._compute_distillation_loss(layer_features, teacher_features, mask, temperature)
            result['distill_loss'] = distill_loss
            total_loss = (1 - distill_weight) * denoise_loss + distill_weight * distill_loss + 0.1 * aux_loss
            result['total_loss'] = total_loss
        else:
            result['total_loss'] = denoise_loss + 0.1 * aux_loss
        batch_features = []
        batch_attentions = []
        for b in range(batch_size):
            batch_features.append([f[b] for f in layer_features])
            batch_attentions.append([a[b] if a is not None else None for a in layer_attentions])
        result['layer_features'] = batch_features
        result['layer_attentions'] = batch_attentions
        return result

    def _compute_distillation_loss(self, student_feats: List[torch.Tensor], teacher_feats: List, mask: Optional[torch.Tensor], temperature: float) -> torch.Tensor:
        """_compute_distillation_loss helper."""
        total_loss = 0.0
        count = 0
        n_student = len(student_feats)
        if len(teacher_feats) > 0 and isinstance(teacher_feats[0], list):
            n_teacher = len(teacher_feats[0])
            batch_size = len(teacher_feats)
            teacher_layers = []
            for layer_idx in range(n_teacher):
                layer_batch = torch.stack([teacher_feats[b][layer_idx] for b in range(batch_size)])
                teacher_layers.append(layer_batch)
        else:
            teacher_layers = teacher_feats
            n_teacher = len(teacher_layers)
        if n_teacher > 0:
            teacher_indices = torch.linspace(0, n_teacher - 1, n_student).long().tolist()
            for s_idx, t_idx in enumerate(teacher_indices):
                s_feat = student_feats[s_idx]
                t_feat = teacher_layers[t_idx]
                if self.feature_aligner is not None:
                    s_feat = self.feature_aligner(s_feat)
                elif s_feat.shape[-1] != t_feat.shape[-1]:
                    min_dim = min(s_feat.shape[-1], t_feat.shape[-1])
                    s_feat = s_feat[..., :min_dim]
                    t_feat = t_feat[..., :min_dim]
                s_feat = s_feat / temperature
                t_feat = t_feat.detach() / temperature
                if mask is not None:
                    s_feat = s_feat * mask.unsqueeze(-1)
                    t_feat = t_feat * mask.unsqueeze(-1)
                loss = F.mse_loss(s_feat, t_feat)
                total_loss += loss
                count += 1
        return total_loss / count if count > 0 else torch.tensor(0.0, device=student_feats[0].device)

    def forward_no_distill(self, atom_types, positions, mask=None):
        """forward_no_distill helper."""
        pred_positions, _ = self.forward(atom_types, positions, mask, return_fixed_dim=False)
        return pred_positions

    def get_fixed_dim_embedding(self, atom_types, positions, mask=None):
        """get_fixed_dim_embedding helper."""
        return self.forward(atom_types, positions, mask, return_fixed_dim=True)
if __name__ == '__main__':
    print('=' * 60)
    print('SE3TransformerStudent')
    print('=' * 60)
    batch_size = 4
    n_atoms = 20
    n_atom_types = 10
    print('\n[1] output_dim=None')
    model1 = SE3TransformerStudent(n_atom_types=n_atom_types, dim=64, output_dim=None, pooling_type='attention')
    atom_types = torch.randint(0, n_atom_types, (batch_size, n_atoms))
    clean_pos = torch.randn(batch_size, n_atoms, 3)
    noisy_pos = clean_pos + 0.1 * torch.randn_like(clean_pos)
    mask = torch.ones(batch_size, n_atoms)
    output1 = model1.forward_with_aux_loss(atom_types, noisy_pos, clean_pos, mask)
    print(f"  Graph embedding shape: {output1['graph_embedding'].shape}")
    print(f"  Denoise loss: {output1['denoise_loss'].item():.4f}")
    print(f"  Auxiliary loss: {output1['aux_loss'].item():.4f}")
    print('\n[2] output_dim=256')
    model2 = SE3TransformerStudent(n_atom_types=n_atom_types, dim=64, output_dim=256, pooling_type='attention')
    output2 = model2.forward_with_aux_loss(atom_types, noisy_pos, clean_pos, mask)
    print(f"  Graph embedding shape: {output2['graph_embedding'].shape}")
    print('\n[3] Check output projection gradients')
    model2.zero_grad()
    output2 = model2.forward_with_aux_loss(atom_types, noisy_pos, clean_pos, mask)
    output2['total_loss'].backward()
    proj_grad = model2.output_projection[0].weight.grad
    if proj_grad is not None and proj_grad.abs().sum() > 0:
        print('   Output projection receives gradients.')
    else:
        print('   Warning: output projection did not receive gradients.')
    print('\nStudent model smoke test complete.')
