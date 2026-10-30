"""Evaluator registry for the wipe v2 readout arms (see eval_wipe_v2.py header). W1/W2 change the obs readout, so the
matrix fallback (plain CFMPolicy) cannot load them; they are built with models_v2.load_policy and decoded by the
matrix MatrixPolicy.decode path (fast_native.generate_action_cached; it only calls the model's own
encode_segmentation / encode_state / construct_mask / pos_encoding, which W1/W2 override consistently).
W3 (meta["intent"] = condition, 1x5) needs no entry: the matrix fallback builds IntentCFMPolicy(1, 5, "condition"),
whose state-dict keys are identical to models_v2.DirtIntentCFMPolicy. Noise: native (x0 only) for W1/W2."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from eval_wipe_v2 import register  # noqa: E402
import rollout_eval_matrix as REM  # noqa: E402


def _readout_loader(run_dir, device, ckpt):
    import models_v2 as M2
    pol = REM.MatrixPolicy(run_dir, ckpt=ckpt, device=device, load_model=False)
    assert pol.intent is None
    pol.model = M2.load_policy(run_dir, device, ckpt)
    return pol


register("w1")(_readout_loader)
register("w2")(_readout_loader)
