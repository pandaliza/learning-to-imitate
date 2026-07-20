"""M9 dataset: emit the future-EEF intent target WITHOUT a workspace-latent cache.

M8's dataset path ties the intent target to the w cache — ``mip.datasets.libero_dataset`` only emits
``wsm_intent_target`` inside ``if self.wsm_w_cache_dir:``. M9 has no w cache by construction (that is
the entire point of the arm), so it needs the target on its own. It also wants the h-step TRAJECTORY,
not just the mean over the horizon.

Everything else already exists upstream: the slot-intent task configs set ``intent_conditioning: true``
and ``intent_horizon: 8``, so the base dataset already loads ``eef``, already fits the ``eef``
normalizer, and already widens the sample window (``pad_after``/``dataset_horizon``) enough that the
future slice is guaranteed full-length. So this is a 1-method subclass, not a fork.

The base module is deliberately left untouched — a training run is live against it.
"""
from __future__ import annotations

import numpy as np

from mip.datasets.libero_dataset import LiberoDataset, make_dataset


class IntentFlowDataset(LiberoDataset):
    """LiberoDataset + ``wsm_intent_target``, with no w cache required.

    ``intent_target_mode``:
      * "concat" — the h-step future EEF trajectory, flattened to (h * eef_dim,). Multimodal, so this
        is the target that motivates a flow decoder.
      * "mean"   — the mean future EEF pose over the horizon, (eef_dim,). Matches M8's target exactly;
        use it to isolate the effect of removing w from the effect of changing the target.

    Adds no constructor state beyond ``intent_target_mode``, so :func:`make_intent_flow_dataset` can
    rebind an already-built base instance instead of duplicating the factory's config plumbing.
    """

    intent_target_mode = "concat"

    def sample_to_data(self, sample):
        data = super().sample_to_data(sample)
        # Same normalization as every other eef consumer (train/eval parity).
        eef_normed = self.normalizer["eef"].normalize(sample["eef"].astype(np.float32))
        future_eef = eef_normed[self.obs_steps : self.obs_steps + self.intent_horizon]
        if self.intent_target_mode == "mean":
            target = future_eef.mean(axis=0)                  # (eef_dim,)
        elif self.intent_target_mode == "concat":
            target = future_eef.reshape(-1)                   # (h * eef_dim,)
        else:
            raise ValueError(f"intent_target_mode must be concat|mean, got {self.intent_target_mode!r}")
        data["wsm_intent_target"] = target.astype(np.float32)
        return data


def make_intent_flow_dataset(task_config, intent_target_mode: str = "concat", mode: str = "train"):
    """Build the base LIBERO dataset via the stock factory, then promote it to :class:`IntentFlowDataset`.

    The class swap keeps the factory's config plumbing in ONE place (upstream) rather than copying it
    here to drift. It is sound because the subclass overrides a single method and declares no extra
    instance layout — ``intent_target_mode`` is a class attribute we shadow per-instance below.
    """
    ds = make_dataset(task_config, mode=mode)
    if "eef" not in ds.normalizer:
        raise ValueError(
            "M9 needs the eef normalizer: set intent_conditioning=true (+ intent_horizon) in the task "
            "config. The slot-intent configs already do."
        )
    ds.__class__ = IntentFlowDataset
    ds.intent_target_mode = intent_target_mode
    return ds
