"""Dataset adapters for action-space-identical IDM training.

All adapters expose a common shape::

    {
        "obs_t": Tensor[obs_dim],
        "obs_next": Tensor[obs_dim],
        "action": Tensor[action_dim],   # action_dim == 7 (OSC_POSE)
        "is_contact": bool tensor,
        "idx": int tensor,
        "dataset_id": int tensor,
    }

For low-dim adapters ``obs_*`` are the concatenated proprioceptive state.
For image-feature adapters ``obs_*`` are the precomputed CLS features (e.g.
DINOv2-base 768-D fp16, materialized on disk by ``v2.features``).
"""

from .stats import ActionStats, compute_action_stats, normalize_action
