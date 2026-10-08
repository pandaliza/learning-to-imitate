"""Native parity for square_eval.py (spec section 4).

For each arm, runs the unmodified native path (make_vec_env(seed=SEED0) + collect_rollouts, num_envs=1: the first
native episode starts from the seed-42 reset = manifest reset 0, delta=0) and the square_eval adapter on the
manifest's reset-0/delta-0 state, with identical torch/numpy seeds, and compares every emitted action chunk.

  python examples/openpi/oops/square_parity.py --manifest-dir oops_runs/square_manifest_pv1 --seeds 0,1,2
"""

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oops_common as oc  # noqa: E402
from square_build_manifest import SEED0, load_native, make_single_env  # noqa: E402
from square_eval import ARMS, OneEnvAdapter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--num-steps", type=int, default=9)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    mdir = pathlib.Path(args.manifest_dir)
    cases = oc.read_jsonl(mdir / "cases.jsonl")
    states = np.load(mdir / "init_states.npz")["states"]
    c0 = next(c for c in cases if c["reset_id"] == 0 and c["delta_m"] == 0.0)
    s0 = np.asarray(states[c0["state_row"]])
    report, ok_all = [], True
    for arm, (ckpt, ov) in ARMS.items():
        cdr, cfg = load_native(ov + ["task.num_envs=1", "optimization.device=cpu"])
        dataset = cdr.make_dataset(cfg.task)
        env, lowdim, robo = make_single_env(cfg.task)
        ad = OneEnvAdapter(env, lowdim, robo)
        ad.begin(None)
        cdr.setup_config_for_env(cfg, ad)
        agent, ip = cdr.load_model(str(ckpt), cfg, dataset, "cpu")
        is_fi = getattr(cfg.network, "arch_variant", "flow_action") == "flow_intent"
        if not is_fi:
            agent.config.optimization.sample_mode = "zero"
        for sd in [int(x) for x in args.seeds.split(",")]:
            # native: fresh seeded vector env (its first reset = np.random.seed(SEED0); env.reset())
            envs = cdr.make_vec_env(cfg.task, seed=SEED0)
            oc.seed_everything(sd)
            nat_chunks, nat_succ, nat_obs, _, _ = cdr.collect_rollouts(
                cfg, agent, dataset, envs, n_rollouts=1, device="cpu", is_flow_intent=is_fi,
                intent_predictor=ip, num_steps=args.num_steps)
            envs.close()
            oc.seed_everything(sd)
            ad.begin(s0, c0["native_env_seed"])
            ad_chunks, ad_succ, ad_obs, _, _ = cdr.collect_rollouts(
                cfg, agent, dataset, ad, n_rollouts=1, device="cpu", is_flow_intent=is_fi,
                intent_predictor=ip, num_steps=args.num_steps)
            n = min(len(nat_chunks), len(ad_chunks))
            dif = [float(np.max(np.abs(nat_chunks[k] - ad_chunks[k]))) for k in range(n)]
            odif = float(np.max(np.abs(nat_obs[:n] - ad_obs[:n]))) if nat_obs is not None else None
            first_bad = next((k for k, d in enumerate(dif) if d > args.atol), None)
            ok = first_bad is None and len(nat_chunks) == len(ad_chunks) and nat_succ == ad_succ
            ok_all &= ok
            r = {"arm": arm, "seed": sd, "match": ok, "n_chunks": [len(nat_chunks), len(ad_chunks)],
                 "max_action_diff": max(dif) if dif else None, "first_divergent_chunk": first_bad,
                 "max_obs_diff": odif, "first_obs_diff": float(np.max(np.abs(nat_obs[0] - ad_obs[0]))),
                 "success": [bool(nat_succ[0]), bool(ad_succ[0])]}
            print(json.dumps(r), flush=True)
            report.append(r)
    print("PARITY:", "PASS" if ok_all else "FAIL")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps({"pass": ok_all, "atol": args.atol, "runs": report}, indent=2))


if __name__ == "__main__":
    main()
