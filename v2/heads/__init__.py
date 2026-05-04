"""Inverse-dynamics heads on top of the legacy DirectMLP / RAIDDecoder.

The legacy heads (``DirectMLP``, ``RAIDDecoder``) live in
``v2/legacy/models.py`` and accept ``(obs_t, obs_next[, a_prior])`` of
arbitrary width. This package adds the two new heads from the strategy:
a small causal Transformer over a 4-frame context, and a
Diffusion-Policy IDM with horizon=1.
"""

from .transformer import TransformerIDM
from .diffusion import DiffusionPolicyIDM
from .knn import KNNRetrievalHead
