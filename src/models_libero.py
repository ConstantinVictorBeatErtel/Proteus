"""
Visual RAID decoder architectures for V-JEPA 2 features.

All models operate on frozen 1024-dim V-JEPA 2 features (feat_dim = 1024).
This is the global constant used everywhere for fair comparison.

Contents:
  - DirectMLPVisual:   MLP on (feat_t || feat_next)
  - ConcatMLPVisual:   MLP on (feat_t || feat_next || pooled_retrieved_actions)
  - RAIDDecoderVisual: cross-attention decoder (query = encoded transition,
                        keys/values = encoded retrieved actions)
"""
from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Global constant — used by ALL conditions and the memory bank.
# ---------------------------------------------------------------------------
FEAT_DIM = 1024  # V-JEPA 2 ViT-L output dimension


# ===================================================================
# DirectMLPVisual — no retrieval baseline
# ===================================================================

class DirectMLPVisual(nn.Module):
    def __init__(self, feat_dim: int = FEAT_DIM, action_dim: int = 7,
                 hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, feat_t: torch.Tensor, feat_next: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feat_t, feat_next], dim=-1)
        return self.net(x)


# ===================================================================
# ConcatMLPVisual — retrieval via simple concatenation (fair comparison)
# ===================================================================

class ConcatMLPVisual(nn.Module):
    """
    Retrieval-augmented MLP that mean-pools retrieved actions and concatenates
    them with the transition features.

    This is the FAIR comparison point: same features, same retrieval, but
    a simpler fusion mechanism (no cross-attention). If RAID cross-attn wins,
    the margin is from the attention mechanism, not from retrieval or features.
    """
    def __init__(self, feat_dim: int = FEAT_DIM, action_dim: int = 7,
                 hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.action_dim = action_dim

        # Input: feat_t(1024) || feat_next(1024) || pooled_retrieved(7) = 2055
        in_dim = feat_dim * 2 + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, feat_t: torch.Tensor, feat_next: torch.Tensor,
                retrieved_actions: torch.Tensor) -> torch.Tensor:
        # retrieved_actions: (B, k, 7)
        pooled = retrieved_actions.mean(dim=1)  # (B, 7)
        x = torch.cat([feat_t, feat_next, pooled], dim=-1)
        return self.net(x)


# ===================================================================
# RAIDDecoderVisual — full RAID with cross-attention
# ===================================================================

class RAIDDecoderVisual(nn.Module):
    """
    Retrieval-Augmented Inverse Dynamics decoder using cross-attention.

    Query:       encoded transition (feat_t || feat_next)
    Keys/Values: encoded retrieved actions
    Output:      predicted action (7-dim)

    The cross-attention learns to weight retrieved actions based on how
    relevant they are to the current transition, replacing simple mean-pooling.
    """
    def __init__(
        self,
        feat_dim: int = FEAT_DIM,
        action_dim: int = 7,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.action_dim = action_dim

        in_dim = feat_dim * 2  # 2048

        # Transition encoder: 2048 → 512
        self.transition_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )

        # Action encoder: 7 → 64 (per retrieved action)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
        )

        # Project action encoding to match query dim: 64 → 512
        self.action_proj = nn.Linear(64, hidden_dim)

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            batch_first=True, dropout=0.0,
        )

        # Post-attention head
        self.post_attn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(
        self,
        feat_t: torch.Tensor,
        feat_next: torch.Tensor,
        retrieved_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feat_t:            (B, 1024)
            feat_next:         (B, 1024)
            retrieved_actions: (B, k, 7)

        Returns:
            action_pred: (B, 7)
        """
        B, k, _ = retrieved_actions.shape

        # Query: (B, 1, hidden_dim)
        query = self.transition_encoder(
            torch.cat([feat_t, feat_next], dim=-1)).unsqueeze(1)

        # Keys/Values: (B, k, 64) → (B, k, hidden_dim)
        kv = self.action_proj(self.action_encoder(retrieved_actions))

        attn_out, _ = self.cross_attn(query, kv, kv)  # (B, 1, hidden_dim)
        attn_out = attn_out.squeeze(1)                # (B, hidden_dim)
        return self.post_attn(attn_out)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, D, A, K = 4, FEAT_DIM, 7, 5

    ft  = torch.randn(B, D)
    fn  = torch.randn(B, D)
    ra  = torch.randn(B, K, A)

    dm = DirectMLPVisual(feat_dim=D, action_dim=A)
    out = dm(ft, fn)
    print(f"DirectMLPVisual:    input ({B},{D}) -> {tuple(out.shape)}")

    cm = ConcatMLPVisual(feat_dim=D, action_dim=A)
    out = cm(ft, fn, ra)
    print(f"ConcatMLPVisual:    input ({B},{D})+({B},{K},{A}) -> {tuple(out.shape)}")

    rv = RAIDDecoderVisual(feat_dim=D, action_dim=A)
    out = rv(ft, fn, ra)
    print(f"RAIDDecoderVisual:  input ({B},{D})+({B},{K},{A}) -> {tuple(out.shape)}")

    print(f"\nFEAT_DIM = {FEAT_DIM}")
    print("[models_libero] OK")
