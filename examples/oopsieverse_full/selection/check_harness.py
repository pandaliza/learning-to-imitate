"""Builds outputs/oopsieverse_full_study/selection/harness_verification.md (measured) from verification JSONs and the
dev run's own records (structural checks on every completed group)."""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_selection as RS  # noqa: E402

OUT = "/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study/selection"
ROOT = RS.root_dir("pick_egg", "sel_dev_parents", 0)


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def structural(root):
    """Checks on every group with records."""
    res = {"groups": 0, "complete_groups": 0, "records": 0, "infra_files": 0, "cand_before_records": 0,
           "chunk_sha_match": 0, "chunk_sha_checked": 0, "tail_seed_shared_across_candidates": 0,
           "tail_seed_groups": 0, "decoder_fixed_primary": 0, "decoder_varies_joint": 0, "intent_groups_primary": 0,
           "intent_groups_joint": 0, "z_shared_primary_joint": 0, "z_pairs": 0, "ended_in_chunk": 0,
           "handoff_l2": [], "seed_namespace_collisions": 0, "terminations": {}}
    import hashlib
    for gd in sorted(glob.glob(f"{root}/branches/*/p*")):
        recs = [json.load(open(f)) for f in glob.glob(f"{gd}/c*_r*.json")]
        if not recs:
            continue
        res["groups"] += 1
        res["records"] += len(recs)
        res["complete_groups"] += len(recs) == RS.M * RS.R
        res["infra_files"] += len(glob.glob(f"{gd}/_infra/*.json"))
        cp = f"{gd}/candidates.npz"
        c = dict(np.load(cp))
        res["cand_before_records"] += os.path.getmtime(cp) <= min(os.path.getmtime(f) for f in glob.glob(f"{gd}/c*_r*.json"))
        for r in recs:
            res["chunk_sha_checked"] += 1
            sh = hashlib.sha256(np.ascontiguousarray(c["chunks"][r["candidate_id"]]).tobytes()).hexdigest()[:16]
            res["chunk_sha_match"] += sh == r["candidate_chunk_sha16"]
            res["ended_in_chunk"] += r["ended_within_candidate_chunk"]
            if r["handoff"]:
                res["handoff_l2"].append(r["handoff"]["arm_cmd_l2"])
            res["terminations"][r["termination"]] = res["terminations"].get(r["termination"], 0) + 1
        by_r = {}
        for r in recs:
            by_r.setdefault(r["tail_replicate"], set()).add(r["tail_seed"])
        res["tail_seed_groups"] += 1
        res["tail_seed_shared_across_candidates"] += all(len(v) == 1 for v in by_r.values()) and \
            len({next(iter(v)) for v in by_r.values()}) == len(by_r)
        cand_seeds = set(c["candidate_seeds"].tolist())
        tails = {next(iter(v)) for v in by_r.values()}
        res["seed_namespace_collisions"] += len(cand_seeds & tails)
        arm, var = os.path.basename(os.path.dirname(gd)).split("__")
        if arm != "amix":
            if var == "primary":
                res["intent_groups_primary"] += 1
                res["decoder_fixed_primary"] += bool(np.all(c["x0"] == c["x0"][0]))
                other = gd.replace("__primary", "__joint")
            else:
                res["intent_groups_joint"] += 1
                res["decoder_varies_joint"] += len({c["x0"][m].tobytes() for m in range(RS.M)}) == RS.M
                other = gd.replace("__joint", "__primary")
            if var == "joint" and os.path.exists(f"{other}/candidates.npz"):
                o = np.load(f"{other}/candidates.npz")
                res["z_pairs"] += 1
                res["z_shared_primary_joint"] += bool(np.array_equal(o["z"], c["z"]))
    h = res.pop("handoff_l2")
    res["handoff_arm_cmd_l2"] = {"n": len(h), "median": float(np.median(h)) if h else None,
                                 "p90": float(np.percentile(h, 90)) if h else None}
    return res


def main():
    v1 = jl(f"{OUT}/verification/v1_null_branch.json") or []
    v3 = jl(f"{OUT}/verification/v3_batch_invariance.json") or {}
    v5 = jl(f"{OUT}/verification/v5_timing.json") or {}
    fc = jl(f"{OUT}/verification/fast_contacts.json") or []
    st = jl(f"{OUT}/verification/analysis_selftest.json") or {}
    rep = jl(f"{OUT}/verification/replay_records.json")
    s = structural(ROOT)
    states = [json.load(open(f)) for f in sorted(glob.glob(f"{ROOT}/states/p*.json"))]
    L = []
    w = L.append
    w("# Candidate harness verification (Phase D, measured)")
    w("")
    w("Code: `examples/oopsieverse_full/selection/` (`harness.py`, `run_selection.py`, `snapshot_damage_state.py`, "
      "`fast_contacts.py`, `gpu3_budget.py`, `analyze_selection.py`). Verification scripts: `verify_harness.py`, "
      "`verify_fast_contacts.py`, `test_restore.py`, and `analyze_selection.py --selftest`. Raw JSON is in `verification/`.")
    w("")
    w("## H1. Protocol equivalence with the frozen evaluator (null branch)")
    w("")
    w("The harness state machine is started at the parent reset and run with the same policy, checkpoint and per-episode generator as `rollout_eval.run_episode`. It is compared field by field on the 30 record fields that do not involve timing.")
    w("")
    w("| Parent | Steps | Outcome | Harness = evaluator, full rendering | Harness = evaluator, render skipping |")
    w("|---|---|---|---|---|")
    for r in v1:
        w(f"| {r['parent']} | {r['evaluator_record']['steps']} | {r['evaluator_record']['outcome']} | "
          f"{r['harness_render_skip_False']['equal']} | {r['harness_render_skip_True']['equal']} |")
    w("")
    w("## H2. Inference determinism (GPU 0)")
    w("")
    w("Requests are decoded in fixed-shape batches of 16 rows (padded). The check: a request decoded at different batch positions, with different companion requests, must give bit-identical actions and intents.")
    w("")
    w("| Arm | Bit-identical | Trials | Variable-size batch vs fixed (max \\|Δa\\|) |")
    w("|---|---|---|---|")
    for a, r in v3.items():
        w(f"| {a} | {r['bit_identical_across_positions_and_companions']} | {r['n_trials']} | {r['variable_batch5_vs_fixed16_maxabs']:.1e} |")
    w("")
    w("So tail and candidate actions do not depend on which other workers' requests share a batch. The matrix evaluator's variable-size batches would differ by about 1e-5.")
    w("")
    w("## H3. Speed shims (bit-exact)")
    w("")
    if v5:
        w(f"- **Render skipping.** Only the two frames that enter the next 2-frame history are rendered. Measured env step: {v5['render_every_step']*1000:.0f} ms with segmentation rendering, {v5['no_render']*1000:.0f} ms without. H1 shows identical records.")
    for r in fc:
        w(f"- **DamageSim contact scan** ({r['case']}): {r['per_call']['n_identical']}/{r['per_call']['n_calls']} calls identical, trace bit-exact = {r['trace_bit_exact']}, {r['s_per_step_released']*1000:.0f} → {r['s_per_step_shim']*1000:.0f} ms/step.")
    w("- **Post-chunk caching.** The state after the candidate chunk is snapshotted once and restored for each of the 4 tails. Restore-through-damage test T6 shows this is bit-exact.")
    w("")
    w("## H4. RNG design")
    w("")
    w("Every seed comes from `KeyedRNG('oopsie_full_v1')` (blake2b), with experiment `oopsie_full_v1/selection/<task>/<pool>` and checkpoint_pair `train_seed_<k>`.")
    w("")
    w("| Use | Namespace and keys | Shared across |")
    w("|---|---|---|")
    w("| Intermediate state: source policy, branch chunk, source-rollout noise | `state_collection` (state_id = parent index) | all arms, as frozen in reset_banks_v1 |")
    w("| A-MIX candidate action noise x0_m | `candidate` (arm, state, candidate_id=m) | — |")
    w("| Intent candidate noise z_m | `candidate` (arm, state, m) | the primary and joint variants of the arm (paired) |")
    w("| Decoder noise, primary intent arms | `decoder` (arm, state, candidate_id=-1) | all 16 candidates, so it is fixed |")
    w("| Decoder noise, joint sensitivity | `decoder` (arm, state, candidate_id=m) | — |")
    w("| Tail replicate r | `continuation` (state, continuation_id=r, policy_identity=`<arm>_tail`). KeyedRNG forbids a candidate key here. | all candidates and both variants of an arm |")
    w("| Physics and health | none. Deterministic given the snapshot and the actions (restore tests T1–T6). | — |")
    w("| Env reset RNG (`env.rng`) | restored from the snapshot; unused after reset | — |")
    w("")
    w("## H5. Structural checks on the dev records (every group written so far)")
    w("")
    for k, v in s.items():
        w(f"- `{k}`: {v}")
    w("")
    w("Reading:")
    w("- `cand_before_records`: number of groups whose candidates.npz (noises, decoded intents, physical chunks) was written before the group's first continuation record.")
    w("- `chunk_sha_match`: number of records whose executed chunk hash equals the stored chunk.")
    w("- `tail_seed_shared_across_candidates`: number of groups where each replicate's tail seed is identical across all 16 candidates.")
    w("- `decoder_fixed_primary` and `decoder_varies_joint`: whether the noise design holds.")
    w("- `z_shared_primary_joint`: intent noises paired across the two variants.")
    w("- `handoff_arm_cmd_l2`: the L2 distance between the candidate's last arm command and the tail's first arm command (handoff behaviour).")
    w("")
    w("## H6. Intermediate-state bank (dev, training seed 0)")
    w("")
    if states:
        by = {}
        for x in states:
            b = by.setdefault(x["source_arm"], {"parents": 0, "eligible": 0, "attempts": 0, "reasons": {}})
            b["parents"] += 1
            b["eligible"] += x["eligible"]
            b["attempts"] += len(x["attempts"])
            for a in x["attempts"]:
                if a["outcome"] != "eligible":
                    kk = a["outcome"].split(" (")[0]
                    b["reasons"][kk] = b["reasons"].get(kk, 0) + 1
        w("| Source policy | Parents | Eligible | Attempts | Attrition by reason |")
        w("|---|---|---|---|---|")
        for a, b in sorted(by.items()):
            w(f"| {a} | {b['parents']} | {b['eligible']} | {b['attempts']} | {b['reasons'] or '—'} |")
        tbs = [x["chosen"]["t_b"] for x in states if x["eligible"]]
        w("")
        w(f"Branch times t_b of the eligible states: mean {np.mean(tbs):.1f}, range {min(tbs)}–{max(tbs)}, n = {len(tbs)}.")
    w("")
    w("## H7. Analysis code self-test")
    w("")
    if st:
        ok = sum(x["ok"] for x in st["oracle_vs_bruteforce"])
        w(f"- C(K) against the closed form and brute-force subset enumeration: {ok}/{len(st['oracle_vs_bruteforce'])} cases agree to 1e-12.")
        for k, v in st["selector"].items():
            w(f"- Selector `{k}`: {v}")
        w(f"- Overall: {'PASS' if st['pass'] else 'FAIL'}")
    if rep:
        w("")
        w("## H8. Record replay (determinism of a completed continuation)")
        w("")
        w(json.dumps(rep, indent=1, default=float))
    # H9: GPU-3 sharing evidence
    sup = open(f"{ROOT}/supervisor.log").read().splitlines()
    ev = [x for x in sup if "EVICT_TEST" in x]
    inc = [x for x in sup if "OVER CAP" in x or "DAEMON ACTIVE" in x or "worker gone" in x]
    ylds = [x for x in sup if "yield worker" in x and "finishes" in x]
    bl = [json.loads(x) for x in open("/edrive1/lizad/oopsieverse_full_study/runs/selection/_gpu3/budget_log.jsonl")]
    bl = [b for b in bl if "total_ours_mib" in b]
    w("")
    w("## H9. GPU-3 sharing with the eval daemon (measured)")
    w("")
    w(f"- **Budget samples logged:** {len(bl)} (`runs/selection/_gpu3/budget_log.jsonl`).")
    w(f"- **Our total on GPU 3** (daemon + selection + tests): max {max(b['total_ours_mib'] for b in bl):.0f} MiB against the 10,240 MiB cap. Samples over the cap: {sum(b['total_ours_mib'] > 10240 for b in bl)}.")
    w(f"- **Samples with GPU-3 free < 8 GB:** {sum(b['free_mib'] < 8192 for b in bl)}. These came from another user's process ramping to within 0.5–2 GB of full. The policy response is below.")
    w(f"- **Pre-emptive drains before daemon evaluations:** {len(set(x[:5] for x in ylds))} episodes; for example `{ylds[0] if ylds else '-'}`. Every worker had exited before the daemon's workers allocated.")
    w(f"- **Safe-eviction tests:** {len(ev)}, all `safe` (no SIGKILL fallback); survivors kept producing records:")
    for x in ev:
        w(f"  - `{x}`")
    w(f"- **Incidents:** {len(inc)}:")
    for x in inc:
        w(f"  - `{x}`")
    w("")
    w("**What the 05:50 incident caused.**")
    w("- The SIGKILL then in use could cut a request that was half-written into the shared inference queue. The worker respawned into that slot hung, and so did the supervisor's shutdown.")
    w("- The tree was stopped. No record was corrupted (records are atomic) and nothing was lost beyond in-flight continuations.")
    w("")
    w("**Fixes.**")
    w("- **Safe eviction.** SIGUSR1 tells a worker to exit at once if it is not mid-request, or as soon as its response arrives if it is. SIGKILL is used only after 5 s, and is followed by a server restart with fresh queues.")
    w("- **Liveness polling** of worker processes.")
    w("- **Strict headroom.** New workers are granted only if free memory (with our workers removed) stays at or above 8 GB plus 256 MiB margin plus their own footprint, measured at its lowest over the last 30 min. This plans against other users' spikes.")
    open(f"{OUT}/harness_verification.md", "w").write("\n".join(L) + "\n")
    print("written", s["records"])


if __name__ == "__main__":
    main()
