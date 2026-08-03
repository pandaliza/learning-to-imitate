"""Thin launcher for openpi's stock LIBERO client (external/openpi/examples/libero/main.py) that
first applies the two patches our environment needs but the stock script lacks:

  * mip.envs.libero._robosuite_compat -- shims robosuite 1.5.x back to the LIBERO 1.4 API.
  * torch.load(weights_only=False)    -- LIBERO's *.pruned_init files predate torch's 2.6 default.

Then hands off to the stock main.py unchanged (tyro reads sys.argv), so the eval logic / SR
accounting is openpi's, not ours -- the whole point of using their harness for a stock checkpoint.

  python examples/openpi/_openpi_libero_client.py --args.task-suite-name libero_goal \
      --args.num-trials-per-task 20 --args.host 0.0.0.0 --args.port 8000
"""
import functools
import runpy

import torch

import mip.envs.libero._robosuite_compat  # noqa: F401  (robosuite 1.5 -> libero 1.4 shim)

torch.load = functools.partial(torch.load, weights_only=False)

# robosuite 1.5.x drops the base-env `seed()` method, so LIBERO's ControlEnv.seed() calls
# `self.env.seed(seed)` on a None -> TypeError. Our own eval tolerates this (fixed init states make
# the seed irrelevant); openpi's stock main.py calls it unguarded. Make ControlEnv.seed a safe no-op.
from libero.libero.envs import env_wrapper as _lw  # noqa: E402


def _safe_seed(self, seed=None):
    fn = getattr(self.env, "seed", None)
    if callable(fn):
        try:
            fn(seed)
        except Exception:
            pass


_lw.ControlEnv.seed = _safe_seed

runpy.run_path("external/openpi/examples/libero/main.py", run_name="__main__")
