"""Assemble ``audit_report.md`` (spec §13 item 1) from the saved audit artifacts.

Every number in the report is read from the audit artifacts in ``audit_dir``:

- ``resolved_semantics.json``  resolved checkpoint / policy semantics
- ``rollout_audit.json``       native reproduction, A1 parity, A3 causality, quaternion audit
- ``a2_measurements.json``     A2 restoration tests and ablations
- ``restoration_report.md``    stack versions and the narrative A2 findings (parsed by regex)

Usage::

    .venv/bin/python -m mip.intent_selection.audit_report [--audit-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_DIR = REPO_ROOT / "outputs" / "intent_selection" / "audit"

# The harness evaluates the ACTION checkpoint with Gaussian initial noise. Native
# ACTION evaluation is zero-noise (deterministic), so this arm is labelled non-native.
ACTION_STOCH_ARM = "ACTION_STOCH"


# ============================================================================
# Loading and derived audit facts
# ============================================================================

def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _episodes_from_key(key: str) -> int | None:
    """Parse the episode count from keys like ``intent_envseeds1000-1049``."""
    m = re.search(r"envseeds(\d+)-(\d+)", key)
    if not m:
        return None
    return int(m.group(2)) - int(m.group(1)) + 1


def _pooled(runs: list[tuple[str, float, int]]) -> dict[str, Any]:
    n_success = sum(round(sr * n) for _, sr, n in runs)
    n_episodes = sum(n for _, _, n in runs)
    return {
        "runs": runs,
        "n_success": n_success,
        "n_episodes": n_episodes,
        "rate": n_success / n_episodes if n_episodes else math.nan,
        "min_rate": min(sr for _, sr, _ in runs) if runs else math.nan,
        "max_rate": max(sr for _, sr, _ in runs) if runs else math.nan,
    }


def audit_facts(audit_dir: str | Path = DEFAULT_AUDIT_DIR) -> dict[str, Any]:
    """Read the audit artifacts and derive the facts both reports cite.

    Returns a dict; missing artifacts produce a ``missing`` list instead of an error.
    """
    audit_dir = Path(audit_dir)
    facts: dict[str, Any] = {"audit_dir": str(audit_dir), "missing": []}

    def opt(name):
        p = audit_dir / name
        if not p.exists():
            facts["missing"].append(name)
            return None
        return _load_json(p) if name.endswith(".json") else p.read_text()

    sem = opt("resolved_semantics.json")
    roll = opt("rollout_audit.json")
    a2 = opt("a2_measurements.json")
    rest_md = opt("restoration_report.md")
    facts.update(semantics=sem, rollout=roll, a2=a2, restoration_md=rest_md)

    if roll is not None:
        step0 = roll.get("step0_native", {})
        extra = roll.get("step0_diagnostic_extra_seeds", {})

        def runs_for(step0_key, extra_prefix):
            runs = []
            s0 = step0.get(step0_key)
            if s0:
                runs.append((f"step0 {step0_key} (env seeds "
                             f"{s0['env_seeds'][0]}-{s0['env_seeds'][-1]})",
                             s0["success_rate"], s0["n_episodes"]))
            for k, v in extra.items():
                if k.startswith(extra_prefix) and isinstance(v, (int, float)):
                    n = _episodes_from_key(k)
                    if n:
                        runs.append((k, float(v), n))
            return runs

        facts["sr_intent"] = _pooled(runs_for("intent_native", "intent_envseeds"))
        facts["sr_action_stoch"] = _pooled(runs_for("action_stochastic", "action_stoch_envseeds"))
        facts["sr_action_zero"] = _pooled(runs_for("action_zero_native", "action_zero_envseeds"))
        hist = extra.get("saved_eval_history_mean_success_9", {})
        facts["saved_history"] = {
            arm: {"n_evals": len(v), "mean": sum(v) / len(v), "max": max(v)}
            for arm, v in hist.items() if v
        }
        facts["saved_history_note"] = extra.get("note", "")

    if rest_md is not None:
        m = re.search(r"robosuite ([\d.]+), mujoco ([\d.]+)", rest_md)
        facts["stack_versions"] = {"robosuite": m.group(1), "mujoco": m.group(2)} if m else {}
        m = re.search(r"differ by ([0-9.eE+-]+) over (\d+) steps", rest_md)
        facts["reseed_dev"] = (m.group(1), int(m.group(2))) if m else None
        facts["osc_delta_found"] = 'input_type="delta"' in rest_md
        m = re.search(r"overwrites `qacc_warmstart`\*\* \(measured Δ up to ([0-9.eE+-]+)\)", rest_md)
        facts["warmstart_overwrite"] = m.group(1) if m else None
    return facts


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        if x == 0:
            return "0"
        if math.isinf(x):
            return "inf"
        if abs(x) < 1e-3 or abs(x) >= 1e4:
            return f"{x:.2g}"
        return f"{x:.{nd}f}".rstrip("0").rstrip(".") if nd else f"{x:.0f}"
    return str(x)


def _pf(passed: bool | None) -> str:
    return "PASS" if passed else ("FAIL" if passed is not None else "n/a")


# ============================================================================
# Report sections
# ============================================================================

def _section_semantics(f) -> list[str]:
    sem = f["semantics"]
    if sem is None:
        return ["## 1. Resolved policy semantics", "", "`resolved_semantics.json` missing.", ""]
    ck, obs, act, intent = sem["checkpoints"], sem["observation"], sem["action"], sem["intent"]
    smp, net, env = sem["sampler"], sem["network"], sem["environment"]
    L = ["## 1. Resolved policy semantics", "",
         f"Source: `resolved_semantics.json`, resolved {sem.get('resolution_date', 'n/a')}.", "",
         "| Field | ACTION checkpoint | INTENT checkpoint |", "|---|---|---|"]
    for key, label in [("path", "Path"), ("sha256", "SHA-256"), ("training_step", "Training step"),
                       ("mean_success_9", "Saved success (NFE 9, best-of-history)"),
                       ("mean_action_spread_9", "Saved action spread (NFE 9)")]:
        a, i = ck["action"].get(key), ck["intent"].get(key)
        if key in ("path", "sha256"):
            a, i = f"`{a}`", f"`{i}`"
        else:
            a, i = _fmt(a, 4), _fmt(i, 4)
        L.append(f"| {label} | {a} | {i} |")
    L += ["", "| Semantics | Value |", "|---|---|",
          f"| Observation keys (order) | {', '.join(f'`{k}`' for k in obs['obs_keys'])} |",
          f"| Observation dim / history length | {obs['obs_dim']} / {obs['obs_steps']} |",
          f"| Action dim (internal) / prediction horizon / execution horizon | "
          f"{act['act_dim']} / {act['horizon']} / {act['act_steps']} |",
          f"| abs_action flag | {act['abs_action']} (see limitation L1: the OSC actually runs in delta mode) |",
          f"| Action normalizer | {act['action_normalizer']['type']}, blake2b `{act['action_normalizer']['hash_blake2b']}` |",
          f"| Intent dim / horizon / type | {intent['intent_dim']} / {intent['intent_horizon']} / `{intent['intent_type']}` |",
          f"| Sampler NFE | {smp['nfe']} ({sem.get('nfe_evidence', '')}) |",
          f"| ACTION native sample mode | `{smp['action_sample_mode_native']}` |",
          f"| INTENT intent sample mode | `{smp['intent_sample_mode']}` |",
          f"| Network | {net['network_type']}, emb_dim {net['emb_dim']}, {net['num_layers']} layers, EMA {net['ema_rate']} |",
          f"| Environment | {env['env_name']} / {env['env_type']}, max_episode_steps {env['max_episode_steps']} |"]
    roll = f["rollout"]
    if roll is not None:
        sl = roll.get("step0_native", {}).get("strict_load", {})
        L += ["", "Strict checkpoint load (missing / unexpected keys per module, from `rollout_audit.json`):", "",
              "| Arm | Overrides | Modules | Missing | Unexpected |", "|---|---|---|---|---|"]
        for arm, d in sl.items():
            mods = d.get("modules", {})
            miss = sum(len(v.get("missing", [])) for v in mods.values())
            unexp = sum(len(v.get("unexpected", [])) for v in mods.values())
            L.append(f"| {arm} | `{d.get('overrides', '')}` | {', '.join(mods)} | {miss} | {unexp} |")
    L += ["", "`adapters_audit.md` in the same directory predates this resolution (it lists NFE 1). "
          "`resolved_semantics.json` supersedes it.", ""]
    return L


def _section_native(f) -> list[str]:
    roll = f["rollout"]
    if roll is None:
        return ["## 2. Native evaluation and A1 parity", "", "`rollout_audit.json` missing.", ""]
    step0 = roll["step0_native"]
    L = ["## 2. Native evaluation reproduction and A1 parity", "",
         "### 2.1 Native reproduction (original evaluator path)", "",
         "| Arm | Sample mode | Episodes (env seeds) | Success | Saved checkpoint value |", "|---|---|---|---|---|"]
    saved = {"action": f["semantics"]["checkpoints"]["action"]["mean_success_9"] if f["semantics"] else None,
             "intent": f["semantics"]["checkpoints"]["intent"]["mean_success_9"] if f["semantics"] else None}
    for key, label in [("action_zero_native", "ACTION (native)"),
                       ("action_stochastic", f"{ACTION_STOCH_ARM} (non-native)"),
                       ("intent_native", "INTENT (native)")]:
        d = step0.get(key)
        if not d:
            continue
        sv = _fmt(saved[d["arm"]], 2) if key != "action_stochastic" else "n/a"
        L.append(f"| {label} | `{d['sample_mode']}` | {d['n_episodes']} ({d['env_seeds'][0]}-{d['env_seeds'][-1]}) | "
                 f"{_fmt(d['success_rate'], 2)} ({d['n_success']}/{d['n_episodes']}) | {sv} |")
    L += ["", "Pooled over every audit run (step 0 plus the extra-seed diagnostics):", "",
          "| Arm | Runs | Pooled success | Range across runs |", "|---|---|---|---|"]
    for key, label in [("sr_action_zero", "ACTION (native, zero noise)"),
                       ("sr_action_stoch", f"{ACTION_STOCH_ARM} (non-native)"),
                       ("sr_intent", "INTENT (native)")]:
        p = f.get(key)
        if p and p["n_episodes"]:
            L.append(f"| {label} | {len(p['runs'])} | {_fmt(p['rate'], 3)} ({p['n_success']}/{p['n_episodes']}) | "
                     f"{_fmt(p['min_rate'], 2)}-{_fmt(p['max_rate'], 2)} |")
    a1 = roll.get("a1_parity", {})
    L += ["", "### 2.2 A1 parity: native wrapper vs harness adapter", "",
          f"Lockstep rollouts with identical env seeds and injected noise ({a1.get('noise', '')}). "
          f"Native path: {a1.get('native_wrapper', '')}. Observation tolerance {_fmt(a1.get('tolerance_obs'))}.", "",
          "| Arm | Sample mode | Episodes | Max abs diff: normalized / physical 10-D / env 7-D / obs | Native SR | Adapter SR | Outcomes identical | Verdict |",
          "|---|---|---|---|---|---|---|---|"]
    for key in ("action_zero", "action_stochastic", "intent"):
        d = a1.get(key)
        if not d:
            continue
        mx = d["max_abs_diff"]
        L.append(f"| {key} | `{d['sample_mode']}` | {len(d.get('episodes', []))} | "
                 f"{_fmt(mx['normalized'])} / {_fmt(mx['physical10'])} / {_fmt(mx['env7'])} / {_fmt(mx['obs'])} | "
                 f"{_fmt(d['native_sr'], 2)} | {_fmt(d['adapter_sr'], 2)} | {_fmt(d['success_identical'])} | {_pf(d['pass'])} |")
    L.append("")
    return L


def _section_restoration(f) -> list[str]:
    a2 = f["a2"]
    if a2 is None:
        return ["## 3. Restoration checks (A2)", "", "`a2_measurements.json` missing.", ""]
    tol = a2.get("_tolerance", {})
    sv = f.get("stack_versions") or {}
    L = ["## 3. Restoration checks (A2)", "",
         f"Stack: robosuite {sv.get('robosuite', 'n/a')}, mujoco {sv.get('mujoco', 'n/a')}. "
         f"Required tolerance: {tol.get('required', 'n/a')} (spec ceiling {_fmt(tol.get('ceiling'))}). "
         "Full component list: `restoration_report.md`.", "",
         "| Test | Result | Max abs deviation | Detail |", "|---|---|---|---|"]
    t = a2
    rows = [
        ("1. snapshot, legal synthetic actions, restore, repeat", t.get("test1_synthetic"),
         lambda d: f"{d['steps']} steps, {len(d['per_quantity'])} quantities compared, bitwise={_fmt(d['bitwise'])}"),
        ("1b. same, demo actions through success", t.get("test1b_demo_replay"),
         lambda d: f"{d['steps']} steps, success reached={_fmt(d['success_reached'])} at step {d['first_success_step']}"),
        ("2. restore the same snapshot twice", t.get("test2_restore_twice"),
         lambda d: f"bitwise={_fmt(d['bitwise'])}"),
        ("3. permuted evaluation order", t.get("test3_permutation"),
         lambda d: f"{d['n_keys']} keys, {d['distinct_trajectories']} distinct trajectories"),
        ("4. disk round-trip, subprocess workers, different orders", t.get("test4_disk_subprocess_workers"),
         lambda d: f"{len(d['worker_pids'])} processes, {d['n_comparisons']} comparisons"),
        ("5. native replanning boundary", t.get("test5_replan_boundary"),
         lambda d: (f"elapsed {d['elapsed_steps']}, history equal={_fmt(d['history_equal'])}, "
                    f"not padded={_fmt(d['history_not_padded'])}, pending actions={d['pending_actions']}")),
        ("6. controller A/B/A", t.get("test6_controller_ABA"),
         lambda d: (f"B differs={_fmt(d['B_differs'])}, digest components differing after restore: "
                    f"{len(d['state_digest_components_differing'])}")),
        ("7. transition determinism", t.get("test7_transition_determinism"),
         lambda d: (f"{d['repeats']} repeats, stepping consumes RNG={_fmt(d['stepping_consumes_rng'])}, "
                    f"post-chunk caching valid={_fmt(d['post_chunk_caching_valid'])}")),
    ]
    n_pass = 0
    for label, d, detail in rows:
        if d is None:
            L.append(f"| {label} | missing | n/a | n/a |")
            continue
        n_pass += bool(d.get("passed"))
        L.append(f"| {label} | {_pf(d.get('passed'))} | {_fmt(d.get('max_abs_dev'))} | {detail(d)} |")
    L += ["", f"{n_pass}/{len(rows)} restoration tests pass.", ""]
    abl = a2.get("ablation_missing_component", {})
    if abl:
        L += ["Ablation: leaving one component stale (max abs deviation over the replay window). "
              "A nonzero value shows that the component must be restored:", "",
              "| Component left stale | Max abs dev | qpos dev | obs dev | Steps run |", "|---|---|---|---|---|"]
        for k, d in sorted(abl.items(), key=lambda kv: -(kv[1].get("max_abs_dev", 0) if kv[0][0] != "_" else -1)):
            if k.startswith("_"):
                continue
            L.append(f"| `{k}` | {_fmt(d['max_abs_dev'])} | {_fmt(d['qpos'])} | {_fmt(d['obs'])} | {d['steps_run']} |")
        L.append("")
        stale = abl.get("_stale_vs_snapshot")
        if stale:
            L += [f"The stale values came from step {stale['elapsed_steps_stale']}, the snapshot from step "
                  f"{stale['elapsed_steps_snapshot']}.", ""]
    if f.get("warmstart_overwrite"):
        L += [f"Restore never steps physics. In this mujoco version `mj_forward` overwrites `qacc_warmstart` "
              f"(measured up to {f['warmstart_overwrite']}), so the restore re-applies the integration state "
              "after the forward pass.", ""]
    return L


def _section_causality(f) -> list[str]:
    roll = f["rollout"]
    if roll is None or "a3_intent_causality" not in roll:
        return ["## 4. Intent causality (A3)", "", "A3 results missing.", ""]
    a3 = roll["a3_intent_causality"]
    L = ["## 4. Intent causality (A3)", "",
         f"Intent layout: {a3.get('intent_layout', '')}. {len(a3['states'])} pilot states, 16 keyed intents each.", "",
         "| State t | Intent mean std (norm) | Max pairwise intent pos L2 (m) | Decoded quat norm range | "
         "Pearson / Spearman (intent dist vs action dist) | Manipulated intent pos L2 (m) -> action chunk L2 (phys) | "
         "Same intent twice: max abs | Distinct noise hashes | Batch-16 vs singles max abs |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s in a3["states"]:
        sp, bc, cm, bd = s["a_intent_spread"], s["b_corr"], s["c_manipulation"], s["d_batch_vs_single"]
        qn = sp["decoded_quat_norm_min_max"]
        L.append(f"| {s['state_t']} | {_fmt(sp['norm_mean_std'], 4)} | {_fmt(sp['max_pairwise_pos_l2_m'], 4)} | "
                 f"{qn[0]:.3f}-{qn[1]:.3f} | {_fmt(bc['pearson_norm'])} / {_fmt(bc['spearman_norm'])} | "
                 f"{_fmt(cm['intent_pos_l2_m'], 4)} -> {_fmt(cm['action_chunk_l2_phys10'])} | "
                 f"{_fmt(cm['same_intent_twice_maxabs'])} | {bd['distinct_noise_hashes']} | {_fmt(bd['action_norm_maxabs'], 4)} |")
    pooled = a3.get("b_corr_pooled_norm", {})
    sp_all = [s["a_intent_spread"]["max_pairwise_pos_l2_m"] for s in a3["states"]]
    act_all = [s["c_manipulation"]["action_chunk_l2_phys10"] for s in a3["states"]]
    same = [s["c_manipulation"]["same_intent_twice_maxabs"] for s in a3["states"]]
    hashes = [s["d_batch_vs_single"]["distinct_noise_hashes"] for s in a3["states"]]
    L += ["",
          f"- Conditioning is used: swapping between the 5th and 95th percentile intent changes the action chunk by "
          f"{_fmt(min(act_all))}-{_fmt(max(act_all))} (physical L2) at every state, and the same intent decoded "
          f"twice gives max abs difference {_fmt(max(same))}.",
          (f"- No shared noise across the batch: {min(hashes)} distinct noise hashes per 16 candidates at every state."
           if min(hashes) == max(hashes) else
           f"- Distinct noise hashes per 16 candidates range {min(hashes)}-{max(hashes)}; values below 16 mean shared noise."),
          f"- Intent samples are not collapsed, but they are tightly concentrated: the maximum pairwise position spread "
          f"is {_fmt(min(sp_all) * 1000, 1)}-{_fmt(max(sp_all) * 1000, 1)} mm per state.",
          f"- Intent distance predicts action distance within a state (per-state Pearson "
          f"{_fmt(min(s['b_corr']['pearson_norm'] for s in a3['states']))}-"
          f"{_fmt(max(s['b_corr']['pearson_norm'] for s in a3['states']))}), but only weakly when pooled across states "
          f"(Pearson {_fmt(pooled.get('pearson'))}, Spearman {_fmt(pooled.get('spearman'))}, n={pooled.get('n_pairs')}).",
          "- Leakage checks (from `rollout_audit.json`):"]
    for k, v in a3.get("e_leakage", {}).items():
        L.append(f"  - `{k}`: {v}")
    pc = roll.get("a3_d_precision_check")
    if pc:
        L += ["", "Batch-of-16 vs 16 single calls, by matmul precision (max abs over states):", "",
              "| Device / precision | Intent | Action (normalized) |", "|---|---|---|"]
        for k, rows in pc.items():
            L.append(f"| {k} | {_fmt(max(r['intent_maxabs'] for r in rows), 4)} | "
                     f"{_fmt(max(r['action_maxabs'] for r in rows), 4)} |")
    L.append("")
    return L


def _section_quat(f) -> list[str]:
    roll = f["rollout"]
    q = roll.get("quat_representation") if roll else None
    if not q:
        return ["## 5. Quaternion audit", "", "Missing.", ""]
    mq = q["mean_quat_norm_quantiles(min,1%,50%,max)"]
    rq = q["raw_quat_norm_min_max"]
    return ["## 5. Training intent quaternion audit", "",
            f"{q['note']}", "",
            "| Quantity | Value |", "|---|---|",
            f"| Frames / episodes | {q['n_frames']} / {q['n_episodes']} |",
            f"| Consecutive-frame sign flips (q·q_prev < 0) | {q['consecutive_sign_flips']} |",
            f"| 8-step windows with mixed hemisphere | {q['windows_with_mixed_hemisphere']} / {q['windows']} |",
            f"| Raw quaternion norm min / max | {rq[0]:.9f} / {rq[1]:.9f} |",
            f"| Averaged-quaternion norm (min, 1%, median, max) | {', '.join(f'{v:.4f}' for v in mq)} |",
            "",
            "Because no window mixes hemispheres, averaging without sign alignment does not cancel quaternions "
            f"in this dataset. The averaged label is not unit-norm (minimum {mq[0]:.4f}); the harness does not "
            "renormalize, sign-flip or re-average decoded intents, matching training.", ""]


def _section_limitations(f) -> list[str]:
    sem, roll = f["semantics"], f["rollout"]
    L = ["## 6. Unresolved limitations", "",
         "These do not block the run. Each one limits how far the results can be read.", ""]
    # L1 delta OSC
    abs_flag = sem["action"]["abs_action"] if sem else "n/a"
    L += [f"**L1. The OSC runs in delta mode although `abs_action={str(abs_flag).lower()}`.** "
          "`make_env` sets a top-level `control_delta=False`, but robosuite reads "
          "`body_parts.right.control_delta`, which stays true"
          + (", so the OSC runs with `input_type=\"delta\"`" if f.get("osc_delta_found") else "")
          + ". The dataset actions are delta actions too, so training and evaluation agree. "
          "The `abs_action` flag only selects the rotation conversion (rotvec to rot6d). "
          "Reports must not describe the actions as absolute poses.", ""]
    # L2 reset
    rd = f.get("reset_dev") or f.get("reseed_dev")
    L += ["**L2. A repeated `reset(seed=s)` on the same env is not a true reset.** `RobomimicLowdimWrapper` restores "
          "only qpos/qvel from its `seed_state_map`. Timestep, gripper `current_action`, warmstart and controller "
          "state carry over from the previous episode"
          + (f": the first episode and the cached re-seed differ by {rd[0]} over {rd[1]} steps, while a fresh env "
             "with the same seed matches exactly" if rd else "")
          + ". State-bank snapshots must be taken right after the first seeded reset of a fresh env.", ""]
    # L3 precision
    if roll and roll.get("a3_d_precision_check"):
        pc = roll["a3_d_precision_check"]
        hi = max(r["action_maxabs"] for r in pc.get("cuda_high", [{"action_maxabs": math.nan}]))
        hst = max(r["action_maxabs"] for r in pc.get("cuda_highest", [{"action_maxabs": math.nan}]))
        prec = (f" Measured batch-of-16 vs single-call action differences (normalized): {_fmt(hi, 4)} under "
                f"`high` (TF32) and {_fmt(hst, 4)} under `highest`.")
    else:
        prec = ""
    L += ["**L3. Matmul precision deviates from native evaluation (TF32 `high` to `highest`).** Native "
          "training and evaluation (`train_robomimic.py`) set `torch.set_float32_matmul_precision(\"high\")`, "
          "which allows TF32 on CUDA. The rollout audit used `high`. The harness uses `highest`, so that a "
          "candidate does not depend on its batch size or position (spec A2 test 4)." + prec +
          " Sampled candidates therefore differ slightly from native-precision samples, and the native SR "
          "reproduction was not repeated under `highest`.", ""]
    # L4 saved SR inflation
    sh = f.get("saved_history", {})
    sri = f.get("sr_intent")
    if sh.get("intent") and sri:
        L += [f"**L4. The INTENT saved success rate is inflated by selection.** The checkpoint's saved "
              f"{_fmt(sh['intent']['max'], 2)} is the maximum over {sh['intent']['n_evals']} periodic evaluations "
              f"(mean {_fmt(sh['intent']['mean'], 3)}); the checkpoint is saved on a new best. The pooled audit "
              f"success rate is {_fmt(sri['rate'], 3)} ({sri['n_success']}/{sri['n_episodes']}). The ACTION "
              f"history is also a maximum ({_fmt(sh['action']['max'], 2)} over {sh['action']['n_evals']} evaluations, "
              f"mean {_fmt(sh['action']['mean'], 3)})." if sh.get("action") else "", ""]
    # L5 ACTION_STOCH non-native
    sas, saz = f.get("sr_action_stoch"), f.get("sr_action_zero")
    spread0 = None
    if roll and "step1_spread" in roll:
        spread0 = roll["step1_spread"].get("action_zero", {}).get("avg_max_pairwise_l2_physical10")
    L += [f"**L5. Native ACTION is deterministic, so {ACTION_STOCH_ARM} is a non-native arm.** The ACTION "
          f"checkpoint's native sample mode is `{sem['sampler']['action_sample_mode_native'] if sem else 'n/a'}`. "
          f"Sixteen native candidates are identical (mean max pairwise L2 {_fmt(spread0)}), so the arm cannot "
          f"supply candidates. {ACTION_STOCH_ARM} samples the same checkpoint from Gaussian initial noise"
          + (f"; its pooled audit success is {_fmt(sas['rate'], 3)} ({sas['n_success']}/{sas['n_episodes']}, "
             f"runs {_fmt(sas['min_rate'], 2)}-{_fmt(sas['max_rate'], 2)})" if sas and sas["n_episodes"] else "")
          + (f", against {_fmt(saz['rate'], 3)} ({saz['n_success']}/{saz['n_episodes']}) for native zero-noise ACTION"
             if saz and saz["n_episodes"] else "")
          + ". Results compare INTENT against this non-native proposal distribution, not against the ACTION policy "
          "as trained and evaluated.", ""]
    if sas and sri and sas["n_episodes"] and sri["n_episodes"]:
        L += [f"**Checkpoint-quality confounder.** Measured full-episode success is INTENT {_fmt(sri['rate'], 3)} "
              f"against {ACTION_STOCH_ARM} {_fmt(sas['min_rate'], 2)}-{_fmt(sas['max_rate'], 2)}. The arms also "
              "differ in architecture (intent flow plus MLP decoder vs a direct action flow) and training steps, "
              "and there is one checkpoint pair.", ""]
    return L


def build_audit_report(audit_dir: str | Path = DEFAULT_AUDIT_DIR) -> str:
    f = audit_facts(audit_dir)
    L = ["# Audit report: intent selection, robomimic square-mh-state", "",
         f"Generated by `mip/intent_selection/audit_report.py` from the artifacts in `{Path(audit_dir)}`. "
         "Every number below is read from those files.", ""]
    if f["missing"]:
        L += [f"**Missing artifacts:** {', '.join(f['missing'])}", ""]
    L += ["## Summary", ""]
    a2, roll = f["a2"], f["rollout"]
    if a2:
        tests = [k for k in a2 if k.startswith("test")]
        L.append(f"- Restoration (A2): {sum(bool(a2[k].get('passed')) for k in tests)}/{len(tests)} tests pass, "
                 f"maximum deviation {_fmt(max(a2[k].get('max_abs_dev', 0) for k in tests))}.")
    if roll and "a1_parity" in roll:
        a1 = {k: v for k, v in roll["a1_parity"].items() if isinstance(v, dict) and "pass" in v}
        L.append(f"- Native parity (A1): {sum(v['pass'] for v in a1.values())}/{len(a1)} arms pass, "
                 f"maximum action difference {_fmt(max(v['max_abs_diff']['normalized'] for v in a1.values()))}.")
    if roll and "a3_intent_causality" in roll:
        L.append("- Intent causality (A3): the decoder uses its intent input; no shared noise or leakage of "
                 "future observations or ground-truth intent was found.")
    if f.get("sr_intent") and f.get("sr_action_stoch"):
        L.append(f"- Checkpoint quality differs: INTENT {_fmt(f['sr_intent']['rate'], 3)} vs "
                 f"{ACTION_STOCH_ARM} {_fmt(f['sr_action_stoch']['rate'], 3)} pooled audit success.")
    L += ["- Five limitations remain (section 6): delta-mode OSC, imperfect repeated reset, the `highest` "
          f"precision deviation, INTENT saved-SR inflation, and the non-native {ACTION_STOCH_ARM} arm.", ""]
    for sec in (_section_semantics, _section_native, _section_restoration, _section_causality,
                _section_quat, _section_limitations):
        L += sec(f)
    return "\n".join(L).rstrip() + "\n"


def write_audit_report(audit_dir: str | Path = DEFAULT_AUDIT_DIR, out_path: str | Path | None = None) -> Path:
    out = Path(out_path) if out_path else Path(audit_dir) / "audit_report.md"
    out.write_text(build_audit_report(audit_dir))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    print(f"Wrote {write_audit_report(args.audit_dir, args.out)}")


if __name__ == "__main__":
    main()
