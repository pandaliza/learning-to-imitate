"""Shared runtime for the intent-selection harness (config, policies, env, hashing, IO).

Used by state_bank.py, candidates.py, branch_eval.py and examples/intent_selection/run_experiment.py.

Design decisions (frozen in the YAML configs and the manifest):
  * Inference runs on CPU, 1 torch thread per process, float32 matmul precision "highest",
    batch size 1 for every policy call (candidate generation and tails). This makes candidate
    identity and tail actions independent of batch size, worker count and job order. Measured
    (square-mh, 17.6M-param flow MLP, NFE 9): CPU 1 thread ~75 ms/sample, CUDA B=1 ~21 ms; the
    env costs ~18 ms/step (~147 ms per 8-step chunk), so CPU workers are within ~1.5x of GPU
    per replan and avoid sharing a GPU between dozens of processes.
  * Arms: ACTION_STOCH (ACTION checkpoint, Gaussian initial noise; NON-NATIVE: native ACTION
    sample mode is "zero"); INTENT (native intent noise, deterministic MLP decoder).
  * Each arm's tail is its own policy in the same mode (native tails).
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
OUTPUT_ROOT = REPO / "outputs" / "intent_selection"

# Audit (Stage A) env seeds: never reused by the state bank.
AUDIT_SEED_RANGES = [(0, 24), (1000, 1049)]
# Preregistered, disjoint parent-reset seed ranges per phase (inclusive bounds).
PHASE_SEED_RANGES = {
    "test": {"primary": (90_000, 90_099), "reserve": (95_000, 95_099)},
    "smoke": {"primary": (100_000, 100_099), "reserve": (150_000, 150_099)},
    "pilot": {"primary": (200_000, 200_999), "reserve": (250_000, 250_999)},
    "confirmation": {"primary": (300_000, 300_999), "reserve": (350_000, 350_999)},
}

ARM_NAMES = ("ACTION_STOCH", "INTENT")
TAIL_MODES = ("native",)
RECORD_VERSION = "1.0"

# Config keys that do not affect any outcome; excluded from the config hash so a rerun with a
# different worker count / job order / output dir is the same experiment identity.
_NON_IDENTITY_KEYS = {("runtime", "num_workers"), ("runtime", "job_order_seed"),
                      ("runtime", "max_wall_hours"), ("runtime", "max_env_steps"),
                      ("runtime", "resume"), ("output_dir",)}


# ----------------------------------------------------------------------------- hashing
def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def hash_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=1, sort_keys=True, default=_json_default).encode())


def atomic_save_npz(path: str | Path, **arrays) -> None:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    atomic_write_bytes(path, buf.getvalue())


# ----------------------------------------------------------------------------- config
def load_config(path_or_name: str | Path, overrides: Optional[dict] = None) -> dict:
    p = Path(path_or_name)
    if not p.exists():
        p = CONFIG_DIR / f"{path_or_name}.yaml"
    cfg = yaml.safe_load(p.read_text())
    cfg["_config_path"] = str(p)
    if overrides:
        cfg = deep_update(cfg, overrides)
    validate_config(cfg)
    return cfg


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def identity_config(cfg: dict) -> dict:
    out = copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")})
    for path in _NON_IDENTITY_KEYS:
        d = out
        for k in path[:-1]:
            d = d.get(k, {})
        if isinstance(d, dict):
            d.pop(path[-1], None)
    return out


def config_hash(cfg: dict) -> str:
    return sha256_json(identity_config(cfg))[:16]


def output_dir(cfg: dict) -> Path:
    if cfg.get("output_dir"):
        return Path(cfg["output_dir"])
    return OUTPUT_ROOT / cfg["phase"]


def parent_seed_lists(cfg: dict) -> tuple[list[int], list[int]]:
    """Preregistered primary and reserve parent reset seeds for the phase."""
    sb = cfg["state_bank"]
    rng_ = PHASE_SEED_RANGES[cfg["seed_range_phase"]]
    p0, p1 = rng_["primary"]
    r0, r1 = rng_["reserve"]
    primary = list(range(p0, p0 + int(sb["parent_resets"])))
    reserve = list(range(r0, r0 + int(sb["reserve_parents"])))
    assert primary[-1] <= p1 and (not reserve or reserve[-1] <= r1)
    return primary, reserve


def validate_config(cfg: dict) -> None:
    req = ["experiment_id", "phase", "seed_range_phase", "checkpoints", "checkpoint_sha256",
           "checkpoint_pair_id", "execution_horizon", "episode_horizon", "candidate_pool_size",
           "continuation_replicates", "state_bank", "runtime", "arms", "nfe", "inference"]
    missing = [k for k in req if k not in cfg]
    if missing:
        raise ValueError(f"config missing keys {missing}")
    if cfg["seed_range_phase"] not in PHASE_SEED_RANGES:
        raise ValueError(f"unknown seed_range_phase {cfg['seed_range_phase']}")
    # disjointness of all registered ranges, and from audit seeds
    ranges = [r for ph in PHASE_SEED_RANGES.values() for r in ph.values()] + AUDIT_SEED_RANGES
    for i, a in enumerate(ranges):
        for b in ranges[i + 1:]:
            if not (a[1] < b[0] or b[1] < a[0]):
                raise ValueError(f"seed ranges overlap: {a} {b}")
    for v in cfg.values():
        if isinstance(v, str) and ("REQUIRED" in v or "RESOLVE" in v):
            raise ValueError(f"unresolved config value {v}")
    if int(cfg["execution_horizon"]) != 8:
        raise ValueError("H_exec must be 8 for both arms (resolved from checkpoints)")
    if set(cfg["arms"]) != set(ARM_NAMES):
        raise ValueError(f"arms must be {ARM_NAMES}")
    if cfg["phase"] != "test" and int(cfg["episode_horizon"]) != 400:
        raise ValueError("episode_horizon must be the native 400 outside test configs")


# ----------------------------------------------------------------------------- inference setup
def setup_inference_process(cfg: dict) -> None:
    """Pin the numerics of this process (call before building adapters)."""
    import torch

    inf = cfg["inference"]
    if inf["device"] != "cpu":
        raise ValueError("this harness pins CPU inference (see module docstring)")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(int(inf["torch_threads"]))
    torch.set_float32_matmul_precision(inf["float32_matmul_precision"])


def build_hydra_config(cfg: dict, ckpt_key: str):
    """Compose the training config exactly like train_robomimic.py (hydra, examples/configs)."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "examples" / "configs"), version_base=None):
        hc = compose(config_name="main", overrides=list(cfg["hydra_overrides"][ckpt_key]) + [
            f"optimization.device={cfg['inference']['device']}",
            f"optimization.num_steps={int(cfg['nfe'])}",
            "task.save_video=false",
            "log.save_video=false",
            f"task.max_episode_steps={int(cfg['episode_horizon'])}",
        ])
    hc.task.obs_dim = int(cfg["obs_dim"])  # train_robomimic main(): obs_dim from the env (23)
    return hc


_DATASETS: dict = {}


def get_dataset(hc):
    from mip.datasets.robomimic_dataset import make_dataset

    key = (hc.task.env_name, hc.task.env_type, bool(getattr(hc.task, "intent_conditioning", False)))
    if key not in _DATASETS:
        with contextlib.redirect_stdout(io.StringIO()):
            _DATASETS[key] = make_dataset(hc.task)
    return _DATASETS[key]


@dataclass
class ArmPolicy:
    name: str
    adapter: Any
    obs_normalizer: Any
    checkpoint_sha256: str
    identity: str  # policy identity for continuation seeds

    def normalize_history(self, hist: np.ndarray):
        import torch

        return torch.tensor(self.obs_normalizer.normalize(np.asarray(hist).astype(np.float32)),
                            dtype=torch.float32)


def policy_identity(cfg: dict, arm: str) -> str:
    a = cfg["arms"][arm]
    sha = cfg["checkpoint_sha256"][a["checkpoint"]]
    return f"{arm}:{sha[:16]}:{a['sample_mode']}:nfe{int(cfg['nfe'])}:ema{int(bool(cfg['use_ema']))}"


def build_arm_policy(cfg: dict, arm: str, verify_sha: bool = False) -> ArmPolicy:
    from mip.intent_selection.adapters import PolicyAdapter

    a = cfg["arms"][arm]
    ck = a["checkpoint"]
    path = cfg["checkpoints"][ck]
    if verify_sha and sha256_file(path) != cfg["checkpoint_sha256"][ck]:
        raise ValueError(f"checkpoint sha256 mismatch for {path}")
    hc = build_hydra_config(cfg, ck)
    ds = get_dataset(hc)
    is_intent = a["checkpoint"] == "intent"
    with contextlib.redirect_stdout(io.StringIO()):
        ad = PolicyAdapter(path, hc, arm_name="intent" if is_intent else "action",
                           device=cfg["inference"]["device"], use_ema=bool(cfg["use_ema"]),
                           action_normalizer=ds.normalizer["action"],
                           undo_transform_action=ds.undo_transform_action,
                           action_sample_mode=None if is_intent else a["sample_mode"],
                           num_steps=int(cfg["nfe"]))
    for k, v in (("horizon", "prediction_horizon"), ("act_steps", "execution_horizon"),
                 ("obs_steps", "obs_steps")):
        if int(getattr(hc.task, k)) != int(cfg[v]):
            raise ValueError(f"{arm}: task.{k}={getattr(hc.task, k)} != config {v}={cfg[v]}")
    return ArmPolicy(arm, ad, ds.normalizer["obs"]["state"], cfg["checkpoint_sha256"][ck],
                     policy_identity(cfg, arm))


# ----------------------------------------------------------------------------- env
def build_env(cfg: dict, construction_seed: int):
    """make_env(task, idx=0, render=False, seed=construction_seed)() (MultiStepWrapper stack)."""
    from mip.envs.robomimic.robomimic_env import make_env

    hc = build_hydra_config(cfg, "action")
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_env(hc.task, 0, render=False, seed=int(construction_seed))()
    return env


class SimError(RuntimeError):
    pass


class StepProbe:
    """Per-step simulator readouts + MuJoCo instability detection."""

    _WARN = ("mjWARN_BADQACC", "mjWARN_BADQPOS", "mjWARN_BADQVEL")

    def __init__(self, env):
        import mujoco

        from mip.intent_selection.snapshots import find_layers

        self.layers = find_layers(env)
        self.rs = self.layers["robosuite"]
        self.ms = self.layers["multistep"]
        self.robot = self.rs.robots[0]
        self.nut_bid = self.rs.obj_body_id[self.rs.nuts[self.rs.nut_id].name]
        self.eef_sid = self.robot.eef_site_id["right"]
        self.warn_idx = [int(getattr(mujoco.mjtWarning, w)) for w in self._WARN]

    def warn_counts(self):
        d = self.rs.sim.data._data
        return [int(d.warning[i].number) for i in self.warn_idx]

    def readout(self):
        d = self.rs.sim.data
        return (np.array(d.site_xpos[self.eef_sid], dtype=np.float64),
                np.array(d.xpos[self.nut_bid], dtype=np.float64),
                np.array(d.xquat[self.nut_bid], dtype=np.float64))

    @property
    def elapsed(self) -> int:
        return len(self.ms.reward)


def step_one(env, probe: StepProbe, action: np.ndarray):
    """One physical step through the full wrapper stack. Raises SimError on instability."""
    before = probe.warn_counts()
    obs, reward, terminated, truncated, info = env.step(action[None])
    after = probe.warn_counts()
    if after != before:
        raise SimError(f"mujoco instability warnings {before}->{after}")
    if not np.all(np.isfinite(obs)):
        raise SimError("nonfinite observation")
    return obs, float(probe.ms.reward[-1]), bool(probe.ms.done[-1])


# ----------------------------------------------------------------------------- misc
def git_info() -> dict:
    import subprocess

    def run(*a):
        try:
            return subprocess.run(a, cwd=REPO, capture_output=True, check=True).stdout
        except Exception as e:  # pragma: no cover
            return str(e).encode()

    commit = run("git", "rev-parse", "HEAD").decode().strip()
    diff = run("git", "diff", "HEAD")
    untracked = run("git", "ls-files", "--others", "--exclude-standard", "mip/intent_selection",
                    "examples/intent_selection")
    h = hashlib.sha256(diff)
    for rel in sorted(untracked.decode().split()):
        p = REPO / rel
        if p.is_file() and "__pycache__" not in rel:
            h.update(rel.encode())
            h.update(p.read_bytes())
    return {"commit": commit, "dirty_diff_sha256": h.hexdigest(),
            "dirty": bool(diff.strip()) or bool(untracked.strip()),
            "dirty_hash_scope": "git diff HEAD + untracked files under mip/intent_selection and "
                                "examples/intent_selection"}


def state_id_for(parent_reset_id: int, stratum: str) -> int:
    return int(parent_reset_id) * 10 + {"reset": 0, "mid": 1}[stratum]


def stable_unit(experiment_id: str, *parts) -> float:
    """Keyed uniform in [0,1) from a stable hash (no Python hash())."""
    s = "|".join([experiment_id, *map(str, parts)])
    v = int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")
    return v / 2**64


def silence_logs():
    try:
        from loguru import logger

        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    except Exception:
        pass
