"""Distill-Mol module."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class CrossAttention(nn.Module):
    """CrossAttention implementation."""

    def __init__(self, dim: int, num_heads: int=4, dropout: float=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """forward helper."""
        B, D = query.shape
        q = self.q_proj(query).view(B, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).view(B, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).view(B, self.num_heads, self.head_dim)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).view(B, D)
        out = self.out_proj(out)
        return self.norm(query + out)

class DynamicCrossModalFusion(nn.Module):
    """DynamicCrossModalFusion implementation."""

    def __init__(self, modal_dim: int=256, fusion_dim: int=512, num_modals: int=3, num_heads: int=4, dropout: float=0.1):
        super().__init__()
        self.modal_dim = modal_dim
        self.fusion_dim = fusion_dim
        self.num_modals = num_modals
        self.cross_attns = nn.ModuleList()
        for i in range(num_modals):
            modal_cross = nn.ModuleList([CrossAttention(modal_dim, num_heads, dropout) for j in range(num_modals) if j != i])
            self.cross_attns.append(modal_cross)
        self.cross_fusion = nn.ModuleList([nn.Sequential(nn.Linear(modal_dim * (num_modals - 1), modal_dim), nn.LayerNorm(modal_dim), nn.ReLU(), nn.Dropout(dropout)) for _ in range(num_modals)])
        self.self_enhance = nn.ModuleList([nn.Sequential(nn.Linear(modal_dim * 2, modal_dim), nn.LayerNorm(modal_dim), nn.ReLU(), nn.Dropout(dropout)) for _ in range(num_modals)])
        self.gate = nn.Sequential(nn.Linear(modal_dim * num_modals, fusion_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fusion_dim, num_modals), nn.Softmax(dim=-1))
        self.fusion_proj = nn.Sequential(nn.Linear(modal_dim * num_modals, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fusion_dim, fusion_dim), nn.LayerNorm(fusion_dim))

    def forward(self, modal_features: List[torch.Tensor], return_weights: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """forward helper."""
        num_modals = len(modal_features)
        if num_modals == 1:
            feat = modal_features[0]
            if not hasattr(self, 'single_modal_proj'):
                self.single_modal_proj = nn.Sequential(nn.Linear(self.modal_dim, self.fusion_dim), nn.LayerNorm(self.fusion_dim)).to(feat.device)
            fused = self.single_modal_proj(feat)
            if return_weights:
                weights = torch.ones(feat.size(0), 1, device=feat.device)
                return (fused, weights)
            return fused
        enhanced = []
        for i in range(num_modals):
            cross_outs = []
            cross_idx = 0
            for j in range(num_modals):
                if j != i:
                    cross_out = self.cross_attns[i][cross_idx](modal_features[i], modal_features[j])
                    cross_outs.append(cross_out)
                    cross_idx += 1
            cross_concat = torch.cat(cross_outs, dim=-1)
            cross_fused = self.cross_fusion[i](cross_concat)
            combined = torch.cat([modal_features[i], cross_fused], dim=-1)
            enhanced_feat = self.self_enhance[i](combined)
            enhanced.append(enhanced_feat)
        all_concat = torch.cat(enhanced, dim=-1)
        weights = self.gate(all_concat)
        weighted = []
        for i in range(num_modals):
            w = weights[:, i:i + 1]
            weighted.append(enhanced[i] * w)
        weighted_concat = torch.cat(weighted, dim=-1)
        fused = self.fusion_proj(weighted_concat)
        if return_weights:
            return (fused, weights)
        return fused

class GatedFusion(nn.Module):
    """GatedFusion implementation."""

    def __init__(self, modal_dim: int=256, fusion_dim: int=512, num_modals: int=3, dropout: float=0.1):
        super().__init__()
        self.projections = nn.ModuleList([nn.Sequential(nn.Linear(modal_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU(), nn.Dropout(dropout)) for _ in range(num_modals)])
        self.gate = nn.Sequential(nn.Linear(modal_dim * num_modals, fusion_dim), nn.ReLU(), nn.Linear(fusion_dim, num_modals), nn.Softmax(dim=-1))
        self.output = nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.LayerNorm(fusion_dim))

    def forward(self, modal_features: List[torch.Tensor], return_weights: bool=False):
        projected = [proj(feat) for proj, feat in zip(self.projections, modal_features)]
        concat = torch.cat(modal_features, dim=-1)
        weights = self.gate(concat)
        fused = sum((w.unsqueeze(-1) * p for w, p in zip(weights.unbind(dim=-1), projected)))
        fused = self.output(fused)
        if return_weights:
            return (fused, weights)
        return fused

class MultiModalMoleculeModel(nn.Module):
    """MultiModalMoleculeModel implementation."""

    def __init__(self, roberta_path: Optional[str]=None, gat_checkpoint: Optional[str]=None, node_feature_dim_2d: int=130, gat_hidden_dim: int=256, gat_output_dim: int=256, se3_checkpoint: Optional[str]=None, n_atom_types: int=100, se3_dim: int=64, se3_output_dim: int=256, fusion_dim: int=512, fusion_type: str='cross', use_1d: bool=True, use_2d: bool=True, use_3d: bool=True, freeze_pretrained: bool=True, dropout: float=0.1):
        super().__init__()
        self.use_1d = use_1d
        self.use_2d = use_2d
        self.use_3d = use_3d
        self.fusion_dim = fusion_dim
        self.modal_dim = 256
        active_modals = sum([use_1d, use_2d, use_3d])
        if active_modals == 0:
            raise ValueError('At least one modality must be enabled.')
        self.modal_names = []
        print(f"\n{'=' * 60}")
        print('Initializing multimodal molecule backbone')
        print(f"{'=' * 60}")
        if use_1d:
            print('\n[1D] RoBERTa')
            from transformers import RobertaModel
            self.smiles_encoder = RobertaModel.from_pretrained(roberta_path)
            roberta_dim = self.smiles_encoder.config.hidden_size
            self.smiles_proj = nn.Sequential(nn.Linear(roberta_dim, self.modal_dim), nn.LayerNorm(self.modal_dim), nn.ReLU(), nn.Dropout(dropout))
            if freeze_pretrained:
                for p in self.smiles_encoder.parameters():
                    p.requires_grad = False
                print('  Frozen pretrained 1D encoder')
            self.modal_names.append('1D')
            print(f'  Projection: {roberta_dim} -> {self.modal_dim}')
        if use_2d:
            print('\n[2D] GAT')
            from pretrain.pretrain_2D import GraphMaskedAutoencoder
            self.graph_encoder = GraphMaskedAutoencoder(node_feature_dim=node_feature_dim_2d, hidden_dim=gat_hidden_dim, output_dim=gat_output_dim, num_layers=4, num_heads=4, dropout=dropout, mask_ratio=0.15, pooling_type='attention')
            if gat_checkpoint:
                self._load_ckpt(self.graph_encoder, gat_checkpoint, 'GAT')
            if freeze_pretrained:
                for p in self.graph_encoder.parameters():
                    p.requires_grad = False
                print('  Frozen pretrained 2D encoder')
            self.modal_names.append('2D')
            print(f'  Output dim: {gat_output_dim}')
        if use_3d:
            print('\n[3D] EGNN')
            from pretrain.models.student_model import SE3TransformerStudent
            self.conformer_encoder = SE3TransformerStudent(n_atom_types=n_atom_types, dim=se3_dim, output_dim=se3_output_dim, edge_dim=32, n_layers=2, n_heads=4, dropout=dropout, k=12, pooling_type='attention')
            print(f'   3D: dim={se3_dim}, output_dim={se3_output_dim}, pooling=attention')
            if se3_checkpoint:
                self._load_ckpt(self.conformer_encoder, se3_checkpoint, 'EGNN')
            if freeze_pretrained:
                for p in self.conformer_encoder.parameters():
                    p.requires_grad = False
                print('  Frozen pretrained 3D encoder')
            self.modal_names.append('3D')
            print(f'  Output dim: {se3_output_dim}')
        print(f'\n[Fusion] {fusion_type}')
        if fusion_type == 'cross':
            self.fusion = DynamicCrossModalFusion(modal_dim=self.modal_dim, fusion_dim=fusion_dim, num_modals=active_modals, num_heads=4, dropout=dropout)
        else:
            self.fusion = GatedFusion(modal_dim=self.modal_dim, fusion_dim=fusion_dim, num_modals=active_modals, dropout=dropout)
        print(f'  Output dim: {fusion_dim}')
        print(f"\n{'=' * 60}\n")

    def _load_ckpt(self, model, path, name):
        """_load_ckpt helper."""
        if not os.path.exists(path):
            print(f'  {name} checkpoint not found: {path}')
            return
        try:
            print(f'  Loading {name} checkpoint...')
            ckpt = torch.load(path, map_location='cpu')
            state_dict = ckpt.get('model_state_dict', ckpt)
            model_state = model.state_dict()
            matched_state = {}
            expanded_params = []
            mismatched_params = []
            for key, param in state_dict.items():
                if key not in model_state:
                    continue
                ckpt_shape = param.shape
                model_shape = model_state[key].shape
                if ckpt_shape == model_shape:
                    matched_state[key] = param
                elif self._is_expandable_embedding(key, ckpt_shape, model_shape):
                    expanded_param = self._expand_embedding(key, param, model_state[key], ckpt_shape, model_shape)
                    if expanded_param is not None:
                        matched_state[key] = expanded_param
                        expanded_params.append((key, ckpt_shape, model_shape))
                    else:
                        mismatched_params.append((key, ckpt_shape, model_shape))
                else:
                    mismatched_params.append((key, ckpt_shape, model_shape))
            model.load_state_dict(matched_state, strict=False)
            print(f'  Loaded {name} checkpoint')
            print(f'    Matched parameters: {len(matched_state)}/{len(state_dict)}')
            if expanded_params:
                print(f'      Expanded embedding parameters: {len(expanded_params)}')
                for key, ckpt_shape, model_shape in expanded_params:
                    ckpt_n, embed_dim = ckpt_shape
                    model_n, _ = model_shape
                    print(f'      - {key}: [{ckpt_n}, {embed_dim}] -> [{model_n}, {embed_dim}]')
                    print(f'        Copied rows: {ckpt_n}')
                    print(f'        Initialized rows: {model_n - ckpt_n}')
            if mismatched_params:
                print(f'      Skipped mismatched parameters: {len(mismatched_params)}')
                for key, ckpt_shape, model_shape in mismatched_params:
                    print(f'      - {key}: checkpoint={tuple(ckpt_shape)}, model={tuple(model_shape)}')
            loaded = len(matched_state)
            total = len(state_dict)
            print(f'      Load ratio: {loaded / total * 100:.1f}%')
        except Exception as e:
            print(f'  Failed to load {name} checkpoint: {e}')
            import traceback
            traceback.print_exc()

    def _is_expandable_embedding(self, key, ckpt_shape, model_shape):
        """_is_expandable_embedding helper."""
        if 'embedding' not in key.lower() and 'emb' not in key.lower():
            return False
        if len(ckpt_shape) != 2 or len(model_shape) != 2:
            return False
        if ckpt_shape[1] != model_shape[1]:
            return False
        if model_shape[0] <= ckpt_shape[0]:
            return False
        return True

    def _expand_embedding(self, key, ckpt_param, model_param, ckpt_shape, model_shape):
        """_expand_embedding helper."""
        try:
            ckpt_n, embed_dim = ckpt_shape
            model_n, _ = model_shape
            expanded_embedding = torch.zeros(model_n, embed_dim, dtype=ckpt_param.dtype, device=ckpt_param.device)
            expanded_embedding[:ckpt_n] = ckpt_param
            mean = ckpt_param.mean().item()
            std = ckpt_param.std().item()
            new_embeddings = torch.randn(model_n - ckpt_n, embed_dim, dtype=ckpt_param.dtype, device=ckpt_param.device) * std + mean
            expanded_embedding[ckpt_n:] = new_embeddings
            return expanded_embedding
        except Exception as e:
            print(f'       Failed to expand embedding: {e}')
            return None

    def forward(self, batch_data: Dict, return_weights: bool=False, output_attentions: bool=False):
        """forward helper."""
        modal_features = []
        attentions = {}
        if self.use_1d and '1d' in batch_data:
            d = batch_data['1d']
            out = self.smiles_encoder(input_ids=d['input_ids'], attention_mask=d['attention_mask'], output_attentions=output_attentions)
            feat = self.smiles_proj(out.last_hidden_state[:, 0, :])
            modal_features.append(feat)
            if output_attentions and hasattr(out, 'attentions') and (out.attentions is not None):
                attentions['1d_roberta'] = out.attentions
        if self.use_2d and '2d' in batch_data:
            if output_attentions:
                _, feat, gat_attentions = self.graph_encoder(batch_data['2d'], return_embeddings=True, return_attention_weights=True)
                for i, attn_info in enumerate(gat_attentions):
                    attentions[f'gat_layer_{i}'] = attn_info
            else:
                _, feat = self.graph_encoder(batch_data['2d'], return_embeddings=True)
            modal_features.append(feat)
        if self.use_3d and '3d' in batch_data:
            d = batch_data['3d']
            feat = self.conformer_encoder(d['atom_types'], d['positions'], d.get('mask'), return_fixed_dim=True)
            modal_features.append(feat)
        if return_weights:
            fused, weights = self.fusion(modal_features, return_weights=True)
            if output_attentions:
                return (fused, weights, attentions)
            return (fused, weights)
        if output_attentions:
            fused = self.fusion(modal_features)
            return (fused, attentions)
        return self.fusion(modal_features)

class MultiModalPropertyPredictor(nn.Module):
    """MultiModalPropertyPredictor implementation."""

    def __init__(self, multimodal_model: MultiModalMoleculeModel, num_classes: int=2, task_type: str='classification', dropout: float=0.2):
        super().__init__()
        self.backbone = multimodal_model
        self.task_type = task_type
        self.modal_names = multimodal_model.modal_names
        dim = multimodal_model.fusion_dim
        if task_type == 'classification':
            self.head = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim // 2, dim // 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim // 4, num_classes))
        else:
            self.head = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim // 2, dim // 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim // 4, num_classes))

    def forward(self, batch_data, return_weights=False, output_attentions=False):
        """forward helper."""
        if return_weights:
            if output_attentions:
                fused, weights, attentions = self.backbone(batch_data, return_weights=True, output_attentions=True)
                return (self.head(fused), weights, attentions)
            else:
                fused, weights = self.backbone(batch_data, return_weights=True, output_attentions=False)
                return (self.head(fused), weights)
        elif output_attentions:
            fused, attentions = self.backbone(batch_data, return_weights=False, output_attentions=True)
            return (self.head(fused), attentions)
        else:
            fused = self.backbone(batch_data, return_weights=False, output_attentions=False)
            return self.head(fused)
if __name__ == '__main__':
    print('=' * 60)
    print('Fusion module smoke test')
    print('=' * 60)
    B, dim, fusion_dim = (4, 256, 512)
    feat_1d = torch.randn(B, dim)
    feat_2d = torch.randn(B, dim)
    feat_3d = torch.randn(B, dim)
    print('\n[1] DynamicCrossModalFusion')
    fusion = DynamicCrossModalFusion(modal_dim=dim, fusion_dim=fusion_dim, num_modals=3)
    fused, weights = fusion([feat_1d, feat_2d, feat_3d], return_weights=True)
    print(f'    Fused shape: {fused.shape}, weights: {weights[0].tolist()}')
    print('\n[2] GatedFusion')
    gated = GatedFusion(modal_dim=dim, fusion_dim=fusion_dim, num_modals=3)
    fused2, weights2 = gated([feat_1d, feat_2d, feat_3d], return_weights=True)
    print(f'    Fused shape: {fused2.shape}, weights: {weights2[0].tolist()}')
    print('\nSmoke test complete.')
