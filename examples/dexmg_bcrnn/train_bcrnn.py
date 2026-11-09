"""Official DexMimicGen BC-RNN (robomimic) with optional wp4 intent conditioning.

Baseline = the config from external/dexmimicgen_src/.../generate_training_config.py, run with the
robomimic `dexmimicgen` branch named in the DexMimicGen README (cloned to external/robomimic_dexmg and
put first on PYTHONPATH by scripts/dexmg_bcrnn_run.sh; the installed v0.4 has no train.action_keys).
Rollouts are not batched: the 50 rollouts run one at a time.

Intent arm (--intent): the same network plus
  - an intent predictor (2-layer 256-wide SiLU MLP, as MIP's IntentPredictor) that maps the current
    min-max-normalized low-dim obs (eef pos/quat + gripper of both arms) to the 56-D wp4 intent
    (both eef poses at t+2,4,6,8; targets from the sidecar built by make_intent_sidecar.py);
  - the detached prediction appended to the LSTM's per-step input (as MIP's joint mode: the policy
    trains and runs on predicted intent, and the intent loss does not reach the image encoders);
  - loss = BC action loss + MSE(predicted intent, target intent).
Unlike MIP (2-step obs window) the predictor sees only the current step, matching BC-RNN's frame_stack=1.
--intent_key selects which per-step sidecar dataset is the target: `intent` (wp4, default) or e.g.
`hand_intent` (128-D standardized future rendered-hand latents from examples/dexmg_hand_intent/); the
predictor, the detached concatenation and the loss are the same, only the target (and its width) changes.

Usage:
  python train_bcrnn.py --config CONFIG.json --output_dir DIR --seed S [--intent SIDECAR.hdf5]
                        [--intent_key intent|hand_intent|...] [--match_dataset_scene] [--debug] [--resume]
"""

import argparse
import copy
import json
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn

import robomimic.algo.bc as BCMod
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.scripts import train as RobomimicTrain

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)


# --------------------------------------------------------------------------- environments
def patch_env_creation(dataset_path, match_scene):
    """DexMimicGen shims (same as mip/envs/robomimic/robomimic_env.py) for robomimic's train loop."""
    orig = EnvUtils.create_env_from_metadata

    def create(env_meta, *args, **kwargs):
        import dexmimicgen  # noqa: F401  registers the TwoArm* envs

        env_meta = copy.deepcopy(env_meta)
        ek = env_meta["env_kwargs"]
        ek.pop("env_lang", None)  # DexMimicGen envs do not accept env_lang
        ek["use_object_obs"] = False  # datasets carry no object obs
        if os.environ.get("MUJOCO_EGL_DEVICE_ID"):  # dataset pins render GPU 0
            ek["render_gpu_device_id"] = int(os.environ["MUJOCO_EGL_DEVICE_ID"])
        env = orig(env_meta, *args, **kwargs)
        inner = env.env
        get_obs = inner._get_observations

        def get_obs_with_object(*a, **k):  # robomimic reads di["object-state"] unconditionally
            di = get_obs(*a, **k)
            di.setdefault("object-state", np.zeros(0))
            return di

        inner._get_observations = get_obs_with_object
        robomimic_get_obs = env.get_observation

        def get_observation(*a, **k):  # ...but this branch's rollout policy passes every env obs key
            ret = robomimic_get_obs(*a, **k)  # to the network, and zero-size ones break its reshapes
            return {key: v for key, v in ret.items() if np.size(v) > 0}

        env.get_observation = get_observation
        if match_scene:
            from mip.envs.robomimic.robomimic_env import match_dataset_scene

            match_dataset_scene(inner, dataset_path)
        return env

    EnvUtils.create_env_from_metadata = create


# --------------------------------------------------------------------------- intent
def image_feature_slices(group_encoder):
    """(start, end) of every image key's features inside the robomimic encoder's concatenated output
    (ObservationEncoder concatenates per-key features in obs_shapes order; mirrors its output_shape)."""
    enc = group_encoder.nets["obs"]
    slices, start = [], 0
    for k in enc.obs_shapes:
        shape = enc.obs_shapes[k]
        if enc.obs_randomizers[k] is not None:
            shape = enc.obs_randomizers[k].output_shape_in(shape)
        if enc.obs_nets[k] is not None:
            shape = enc.obs_nets[k].output_shape(shape)
        if enc.obs_randomizers[k] is not None:
            shape = enc.obs_randomizers[k].output_shape_out(shape)
        n = int(np.prod(shape))
        if enc.obs_nets[k] is not None:  # image keys have a conv backbone; low-dim keys pass through
            slices.append((start, start + n))
        start += n
    return slices


class IntentObsEncoder(nn.Module):
    """Wraps the RNN's observation encoder: appends the (detached) predicted intent to its features.

    input_mode "proprio": the predictor reads the min-max-normalized low-dim obs (the intent loss does not
    reach the encoder). input_mode "image": it reads only the image-backbone features, NOT detached, so
    the intent loss also trains the image encoders (the target acts as auxiliary supervision of vision).
    Either way the prediction is detached before the RNN, so the action loss never trains the predictor.
    OVERRIDE (eval-only ablation): None | ("shuffle", pool) | ("constant", vec) replaces the prediction
    fed to the RNN with random rows of `pool` (resampled every step) or with the fixed vector `vec`.
    """

    OVERRIDE = None

    def __init__(self, encoder, lowdim_keys, lo, hi, intent_dim, hidden_dim=256, input_mode="proprio"):
        super().__init__()
        self.encoder = encoder
        self.lowdim_keys = list(lowdim_keys)
        self.intent_dim = intent_dim
        self.input_mode = input_mode
        self.register_buffer("lo", torch.as_tensor(lo, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(np.maximum(hi - lo, 1e-6), dtype=torch.float32))
        if input_mode == "image":
            self.img_slices = image_feature_slices(encoder)
            d_in = sum(b - a for a, b in self.img_slices)
        else:
            self.img_slices, d_in = None, len(lo)
        self.predictor = nn.Sequential(
            nn.Linear(d_in, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, intent_dim),
        )
        self.last_pred = None

    def output_shape(self, input_shape=None):
        return [self.encoder.output_shape()[0] + self.intent_dim]

    def forward(self, **inputs):
        feats = self.encoder(**inputs)
        if self.input_mode == "image":
            pred = self.predictor(torch.cat([feats[..., a:b] for a, b in self.img_slices], dim=-1))
        else:
            x = torch.cat([inputs["obs"][k] for k in self.lowdim_keys], dim=-1)
            pred = self.predictor(2 * (x - self.lo) / self.scale - 1)
        self.last_pred = pred
        fed = pred.detach()
        if IntentObsEncoder.OVERRIDE is not None and not self.training:
            kind, val = IntentObsEncoder.OVERRIDE
            if kind == "shuffle":
                idx = torch.randint(0, len(val), fed.shape[:-1], device=val.device)
                fed = val[idx].to(fed)
            else:
                fed = val.to(fed).expand_as(fed)
        return torch.cat([feats, fed], dim=-1)


_BC_RNN = BCMod.BC_RNN


class BC_RNN_Intent(_BC_RNN):
    INTENT = None  # set from the sidecar attrs before the algo is built

    def _create_networks(self):
        super()._create_networks()
        cfg = BC_RNN_Intent.INTENT
        policy = self.nets["policy"]
        enc = policy.nets["encoder"]
        d_feat = enc.output_shape()[0]
        policy.nets["encoder"] = IntentObsEncoder(enc, cfg["lowdim_keys"], cfg["lowdim_lo"],
                                                  cfg["lowdim_hi"], cfg["intent_dim"],
                                                  input_mode=cfg.get("input_mode", "proprio"))
        old = policy.nets["rnn"].nets
        policy.nets["rnn"].nets = type(old)(
            input_size=d_feat + cfg["intent_dim"], hidden_size=old.hidden_size,
            num_layers=old.num_layers, batch_first=True, bidirectional=old.bidirectional,
        )
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        input_batch = super().process_batch_for_training(batch)
        key = BC_RNN_Intent.INTENT["key"]
        input_batch["intent"] = batch["data"][key].to(self.device).float()  # branch nests under "data"
        return input_batch

    def _forward_training(self, batch):
        predictions = super()._forward_training(batch)
        B, T = batch["intent"].shape[:2]
        predictions["intent_pred"] = self.nets["policy"].nets["encoder"].last_pred.reshape(B, T, -1)
        return predictions

    def _compute_losses(self, predictions, batch):
        losses = super()._compute_losses(predictions, batch)
        losses["bc_action_loss"] = losses["action_loss"]
        losses["intent_loss"] = nn.functional.mse_loss(predictions["intent_pred"], batch["intent"])
        # the intent prediction is detached before the RNN, so the two terms train disjoint parameters
        losses["action_loss"] = losses["action_loss"] + losses["intent_loss"]
        return losses

    def log_info(self, info):
        log = super().log_info(info)
        log["BC_Action_Loss"] = info["losses"]["bc_action_loss"].item()
        log["Intent_Loss"] = info["losses"]["intent_loss"].item()
        return log


def setup_intent(sidecar, key, input_mode="proprio"):
    """Read the predictor's normalization/width from the sidecar and swap BC_RNN for BC_RNN_Intent."""
    with h5py.File(sidecar, "r") as f:
        a = f["data"].attrs
        stat = f"{key}_lo" if f"{key}_lo" in a else f"{key}_mean"  # min-max (wp4) or standardized (hand) targets
        BC_RNN_Intent.INTENT = dict(
            key=key,
            lowdim_keys=[k_.decode() if isinstance(k_, bytes) else str(k_) for k_ in a["lowdim_keys"]],
            lowdim_lo=np.asarray(a["lowdim_lo"]), lowdim_hi=np.asarray(a["lowdim_hi"]),
            intent_dim=int(len(a[stat])), input_mode=input_mode,
        )
    BCMod.BC_RNN = BC_RNN_Intent  # algo_config_to_class looks the class up at call time


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--intent", default=None, help="intent sidecar hdf5 (enables the intent arm)")
    p.add_argument("--intent_key", default="intent", help="sidecar per-step dataset used as the intent target")
    p.add_argument("--intent_input", default="proprio", choices=["proprio", "image"],
                   help="predictor input: low-dim obs (default) or the image-backbone features (trains vision)")
    p.add_argument("--match_dataset_scene", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--resume", action="store_true", help="resume from the run's last.pth (robomimic)")
    args = p.parse_args()

    ext = json.load(open(args.config))
    # Batched rollouts need tianshou's SubprocVectorEnv, whose worker processes would not see the env
    # patch below; run the 50 rollouts sequentially instead (same episodes, only slower).
    ext["experiment"]["rollout"]["batched"] = False
    config = config_factory(ext["algo_name"])
    with config.values_unlocked():
        config.update(ext)
    dataset_path = config.train.data
    if isinstance(dataset_path, list):
        dataset_path = dataset_path[0]["path"] if isinstance(dataset_path[0], dict) else dataset_path[0]

    with config.values_unlocked():
        config.train.output_dir = args.output_dir
        config.train.seed = args.seed
        config.experiment.name = os.path.basename(os.path.normpath(args.output_dir))
        if args.intent:
            config.train.data = args.intent
            config.train.dataset_keys = list(config.train.dataset_keys) + [args.intent_key]
        if args.debug:
            config.experiment.epoch_every_n_steps = 10
            config.train.num_epochs = 2
            config.experiment.rollout.n = 2
            config.experiment.rollout.horizon = 20
            config.experiment.rollout.rate = 1
            config.experiment.save.every_n_epochs = 1
            config.train.num_data_workers = 0
            config.train.hdf5_filter_key = None

    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.train_utils as TrainUtils

    def load_dict_from_checkpoint(ckpt_path):
        # PyTorch >= 2.6 defaults torch.load to weights_only=True, which rejects the numpy scalars in
        # robomimic checkpoints. Only this run's own last.pth / model files are loaded through here.
        return torch.load(os.path.expandvars(os.path.expanduser(ckpt_path)), map_location="cpu",
                          weights_only=False)

    FileUtils.load_dict_from_checkpoint = load_dict_from_checkpoint

    if not hasattr(TrainUtils, "SubprocVectorEnv"):
        # rollout_with_stats does isinstance(env, SubprocVectorEnv) even for unbatched rollouts; without
        # tianshou installed the name is undefined. A stand-in class makes that check False.
        TrainUtils.SubprocVectorEnv = type("SubprocVectorEnv", (), {})
    patch_env_creation(dataset_path, args.match_dataset_scene)
    if args.intent:
        setup_intent(args.intent, args.intent_key, args.intent_input)

    config.lock()
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    print(f"[bcrnn] data={config.train.data} intent={bool(args.intent)} intent_key={args.intent_key} "
          f"intent_input={args.intent_input} match_dataset_scene={args.match_dataset_scene} seed={args.seed}",
          flush=True)
    RobomimicTrain.train(config, device=device, resume=args.resume)


if __name__ == "__main__":
    main()
