"""Build and verify the frozen evaluation / selection reset banks (spec sections 7, 10, 11).

Runs in the isolated oopsie env, CPU only, no renderer.

  derive   : keyed seeds (blake2b; mip/intent_selection/rng.py KeyedRNG convention) for every pool
  enumerate: reset EVERY bank seed through the canonical ``seeded_reset`` in long-lived worker envs
             and record an initial-state fingerprint per seed (JSONL on /edrive1)
  tests    : determinism checks on a sample of seeds, each case in a FRESH spawned process:
             T1 fresh(s) vs fresh(s) in two processes
             T2 fresh(s) vs canonical seeded_reset(s) in a used env (enumeration fingerprints) and
                in a used env that was just driven into damage
             T3 a second plain env.reset() on fresh(s) vs the first one
             T4 sticky ep_meta: set_ep_meta(get_ep_meta()) (as teleop/replay do), reseed s2, reset
  Outputs are written by analyze_reset_banks.py.
"""

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RUN_DIR = "/edrive1/lizad/oopsieverse_full_study/runs/reset_banks"

EXPERIMENT_ID = "oopsie_full_v1"
TASKS = ("pick_egg", "wipe_countertop")
# pool -> size. All pools are disjoint per task (checked); parents are shared across training seeds.
POOLS = {
    "eval_dev": 100,            # spec 7/10: 100 development resets (tuning, checkpoint/config choice)
    "eval_test": 200,           # spec 7/10: 200 frozen test reset families
    "sel_dev_parents": 50,      # spec 11: 50 development parent resets per task/training seed
    "sel_test_parents": 100,    # spec 11: 100 fresh test parent resets per task/training seed
    "sel_train_parents": 400,   # spec 11: learned-selection label generation ("new training parents only")
    "reserve": 300,             # replacements / amendments; never used without a logged amendment
}
# Spare key indices per pool, enumerated with the pool. Pre-declared rule (before any policy result):
# a bank entry whose initial state coincides with a demo initial state (analyze_reset_banks.COINCIDENCE_RULE)
# is dropped and the pool is back-filled with the next spare in key order.
SPARES = 5


def load_keyed_rng():
    spec = importlib.util.spec_from_file_location("ks_rng", os.path.join(REPO, "mip/intent_selection/rng.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.KeyedRNG


def derive_bank():
    """Seed = KeyedRNG(EXPERIMENT_ID).derive_seed('env', experiment=f'{EXPERIMENT_ID}/reset_bank/{task}/{pool}', state_id=i).

    That is blake2b(digest_size=8) over '<experiment>|env|state_id=<i>', first 4 bytes little-endian & 0x7FFFFFFF.
    On a (never observed) collision with an earlier seed of the same task, the index is skipped and the
    pool continues at the next index; skipped indices are recorded.
    """
    KR = load_keyed_rng()(EXPERIMENT_ID)
    bank, skipped = {}, []
    for task in TASKS:
        used = set()
        bank[task] = {}
        for pool, n in POOLS.items():
            exp = f"{EXPERIMENT_ID}/reset_bank/{task}/{pool}"
            items, i = [], 0
            while len(items) < n:
                s = KR.derive_seed("env", experiment=exp, state_id=i)
                if s in used:
                    skipped.append(dict(task=task, pool=pool, index=i, seed=s))
                else:
                    used.add(s)
                    items.append(dict(reset_id=f"{task}/{pool}/{len(items):04d}", key_index=i, seed=int(s)))
                i += 1
            spares, j = [], 0
            while len(spares) < SPARES:
                s = KR.derive_seed("env", experiment=exp, state_id=i)
                if s in used:
                    skipped.append(dict(task=task, pool=pool, index=i, seed=s))
                else:
                    used.add(s)
                    spares.append(dict(reset_id=f"{task}/{pool}/spare{j}", key_index=i, seed=int(s)))
                    j += 1
                i += 1
            bank[task][pool] = items
            bank[task][pool + "__spares"] = spares
    return bank, skipped


# ───────────────────────── worker side ─────────────────────────

def _env_setup():
    sys.path.insert(0, HERE)


_W = {}


def _enum_job(args):
    task, rid, seed = args
    _env_setup()
    from reset_probe import fingerprint, make_env, seeded_reset
    try:
        if task not in _W:
            _W[task] = dict(env=make_env(task, seed=None), n=0)
        w = _W[task]
        t0 = time.time()
        obs = seeded_reset(w["env"], seed)
        fp = fingerprint(w["env"], task, obs)
        w["n"] += 1
        fp.update(task=task, reset_id=rid, seed=seed, reset_s=round(time.time() - t0, 2),
                  worker_pid=os.getpid(), worker_reset_index=w["n"])
        return fp
    except Exception as e:
        return dict(task=task, reset_id=rid, seed=seed, error=repr(e), tb=traceback.format_exc()[-1500:])


def _test_case(args):
    """Each call runs in a fresh process (maxtasksperchild=1)."""
    case, task, seed, seed2 = args
    _env_setup()
    from reset_probe import fingerprint, fresh_env_reset, health_info, make_env, seeded_reset
    try:
        out = dict(case=case, task=task, seed=seed, seed2=seed2)
        if case == "fresh":
            env, obs = fresh_env_reset(task, seed)
            out["fp"] = fingerprint(env, task, obs)
        elif case == "fresh_then_plain_reset":
            env, obs = fresh_env_reset(task, seed)
            out["fp_first"] = fingerprint(env, task, obs)
            env.reset()
            out["fp_second"] = fingerprint(env, task, env.get_observations()[0])
        elif case == "damaged_then_seeded":
            env = make_env(task, seed=seed2)
            env.reset()
            a = np.zeros(env.action_dim)
            a[2] = -1.0   # drive the EEF down into the counter/object
            a[6] = 1.0    # close gripper
            for _ in range(40):
                obs, _, _, _ = env.step(a)
            out["before_reset"] = health_info(env, env.get_observations()[0])
            obs = seeded_reset(env, seed)
            out["fp"] = fingerprint(env, task, obs)
            out["after_reset"] = health_info(env, obs)
        elif case == "sticky_ep_meta":
            env = make_env(task, seed=None)
            obs = seeded_reset(env, seed)
            out["fp_first"] = fingerprint(env, task, obs)
            env.set_ep_meta(env.get_ep_meta())   # what DamageDataCollectionWrapper.reset and playback do
            env.rng = np.random.default_rng(seed2)
            env.reset()                           # NOT unset_ep_meta
            out["fp_sticky"] = fingerprint(env, task, env.get_observations()[0])
            out["fp_clean"] = fingerprint(env, task, seeded_reset(env, seed2))
        else:
            raise ValueError(case)
        return out
    except Exception as e:
        return dict(case=case, task=task, seed=seed, seed2=seed2, error=repr(e), tb=traceback.format_exc()[-1500:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["derive", "enumerate", "tests"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--n-test-seeds", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(RUN_DIR, exist_ok=True)
    bank, skipped = derive_bank()
    if a.stage == "derive":
        json.dump(dict(bank=bank, skipped=skipped), open(os.path.join(RUN_DIR, "bank_seeds.json"), "w"), indent=1)
        print({t: {p: len(v) for p, v in d.items()} for t, d in bank.items()}, "skipped", skipped)
        return
    ctx = mp.get_context("spawn")
    if a.stage == "enumerate":
        out = os.path.join(RUN_DIR, "enumeration.jsonl")
        done = set()
        if os.path.exists(out):
            for line in open(out):
                r = json.loads(line)
                if "error" not in r:
                    done.add(r["reset_id"])
        jobs = [(t, it["reset_id"], it["seed"]) for t in TASKS for p in bank[t] for it in bank[t][p] if it["reset_id"] not in done]
        # interleave tasks so each worker builds both envs at most once; chunks keep a worker on one task
        print(f"{len(jobs)} resets, {a.workers} workers", flush=True)
        t0 = time.time()
        with ctx.Pool(a.workers) as pool, open(out, "a") as fo:
            for k, r in enumerate(pool.imap_unordered(_enum_job, jobs, chunksize=8)):
                fo.write(json.dumps(r) + "\n")
                fo.flush()
                if k % 50 == 0 or "error" in r:
                    print(k, r["reset_id"], r.get("reset_s"), r.get("error", ""), f"{time.time() - t0:.0f}s", flush=True)
        return
    # tests: sample seeds spread over the eval pools, deterministic choice
    cases = []
    for t in TASKS:
        sample = [bank[t]["eval_test"][i]["seed"] for i in range(0, 200, 200 // a.n_test_seeds)][: a.n_test_seeds]
        sample += [bank[t]["eval_dev"][0]["seed"], bank[t]["sel_test_parents"][0]["seed"]]
        other = bank[t]["reserve"][-1]["seed"]
        for s in sample:
            cases += [("fresh", t, s, None), ("fresh", t, s, None)]   # two independent processes (T1)
        for s in sample[:4]:
            cases += [("fresh_then_plain_reset", t, s, None), ("damaged_then_seeded", t, s, other)]
        for s in sample[:2]:
            cases += [("sticky_ep_meta", t, s, bank[t]["reserve"][-2]["seed"])]
    out = os.path.join(RUN_DIR, "tests.jsonl")
    print(f"{len(cases)} test cases", flush=True)
    with ctx.Pool(min(a.workers, len(cases)), maxtasksperchild=1) as pool, open(out, "w") as fo:
        for r in pool.imap_unordered(_test_case, cases):
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            print(r["case"], r["task"], r["seed"], r.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
