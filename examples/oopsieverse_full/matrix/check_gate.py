"""Predeclared pass criteria for verification items 2 and 4 (declared 2026-10-06, before the results exist).

Smoke (item 4), for each of the 10 smoke specs (8 arms on pick_egg; amix and ipathg on wipe_countertop):
  S1 the 500-step run is DONE; S2 exactly 5 dev rollout records (eval_dev[0:5], rollout seed 0);
  S3 no infra_error; S4 no invalid (non-finite) action; S5 nfe_per_chunk == 32 for conditioned intent arms, 16 for
  native and AUX arms; S6 conditioned arms log sampled intents. (Task success is NOT required after 500 steps.)
Intervention (item 2), for each pick_egg intent smoke model:
  I1 distinct noise -> distinct intents (min pairwise L2 > 0 over windows) and same noise -> identical intent;
  conditioned arms: I2 decoded actions change when the intent is resampled (spread > 1e-4) and
  I3 directional clamps move the predicted EEF displacement toward the clamp (mean cosine over 6 directions > 0);
  I-PATH-G: I4 clamping gripper tokens high vs low raises the decoded gripper command (mean change > 0);
  AUX-PATH-G: I5 intent clamps/resampling leave the decoded actions exactly unchanged (no conditioning path).
Writes outputs/oopsieverse_full_study/matrix/verification/gate.json; exit 0 iff all pass.

GATE SPLIT (coordinator decision, 2026-10-06 ~21:00 UTC, recorded before any matrix result):
  TRAINING gate  = item 1 (target extraction) + item 3 (bit-identity) + item 2 (intervention I1-I5) + S1 (all 10
                   smoke trainings DONE). Training does not render.
  EVALUATION gate = smoke dev rollouts S2-S6. Must pass before ANY evaluation of matrix checkpoints. Reason: GPU 0
                   (the only authorised GPU) renders corrupt MuJoCo EGL frames and no exact-parity CPU renderer is
                   available yet; the eval path itself was already pixel-verified for A-MIX on GPUs 6/7.
  `python check_gate.py training` exits 0 iff the training gate passes; `python check_gate.py` = both.
"""

import glob
import json
import os
import sys

R = "/edrive1/lizad/oopsieverse_full_study/runs/matrix_smoke"
V = "/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study/matrix/verification"
SPECS = [("pick_egg", a) for a in ("amix", "asafe", "ssmix", "imean", "ipath", "ipathg", "cmean", "auxpathg")] + \
        [("wipe_countertop", "amix"), ("wipe_countertop", "ipathg")]
COND = {"imean", "ipath", "ipathg", "cmean"}
# Amendment 2: smoke rollouts are rendered with EGL on GPU 3 (parity-verified) -> eval_gpu3/
EV = os.environ.get("GATE_SMOKE_EVAL_DIR", "eval_gpu3")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {"smoke": {}, "intervention": {}, "pass": True,
           "gate_split": "training gate = items 1, 3, intervention, smoke trainings; evaluation gate = smoke dev "
                         "rollouts (before ANY evaluation). Reason: GPU 0 EGL frames corrupt; no exact-parity CPU "
                         "renderer yet; eval path pixel-verified for A-MIX earlier (coordinator decision 2026-10-06)."}
    for task, arm in SPECS:
        rid = f"{arm}_{task}_v1_s0__smoke500"
        recs = [json.load(open(f)) for f in sorted(glob.glob(f"{R}/{EV}/{rid}/*__r0.json"))]
        ok = {"S1_done": os.path.exists(f"{R}/{rid}/DONE.json"), "S2_n5": len(recs) == 5,
              "S3_no_infra": all("infra_error" not in r for r in recs),
              "S4_no_invalid": all(not r.get("invalid_action", True) for r in recs),
              "S5_nfe": all(r.get("nfe_per_chunk") == (32 if arm in COND else 16) for r in recs),
              "S6_intents": all(bool(r.get("sampled_intents_first_chunks")) == (arm in COND) for r in recs)}
        ok["outcomes"] = [r.get("outcome") for r in recs]
        ok["steps"] = [r.get("steps") for r in recs]
        ok["pass"] = all(v for k, v in ok.items() if k.startswith("S")) and bool(recs)
        res["smoke"][rid] = ok
    iv = json.load(open(f"{V}/intervention.json")) if os.path.exists(f"{V}/intervention.json") else []
    for r in iv:
        d = r["diversity"]
        c = {"I1": d["all_distinct"] and d["same_noise_identical"]}
        if r["mode"] == "condition":
            c["I2"] = r["resampling"]["action_spread_across_intent_samples_fixed_action_noise"] > 1e-4
            c["I3"] = r["clamp_direction_mean_cos"] > 0
            if r["intent"] == "ipathg":
                c["I4"] = r["gripper_clamp"]["mean_decoded_grip_change_hi_minus_lo"] > 0
        else:
            c["I5"] = (max(v["max_abs_action_change"] for v in r["clamp_direction"].values()) == 0.0 and
                       r["resampling"]["action_spread_across_intent_samples_fixed_action_noise"] == 0.0)
        c["pass"] = all(c.values())
        res["intervention"][r["arm"]] = c
        res["pass"] &= c["pass"]
    if len(iv) != 5:
        res["intervention_missing"] = True
    tl = json.load(open(f"{V}/targets_legacy.json"))
    tg = json.load(open(f"{V}/targets_geometry.json"))
    item1 = (all(c["max_abs_err_same_teleop_source"] < 1e-6 and c["max_abs_err_vs_playback_source (normalised units)"] < 1e-3
                 for t in tl.values() for c in t["conditions"].values())
             and all(g["mat_6d_mat_maxerr"] < 1e-9 and g["rel_compose_pos_maxerr_m"] < 1e-9 and g["targets_finite"]
                     and g["noisy6d_projection_ok"] and g["quat_conv_maxerr_vs_robosuite_quat2mat"] < 1e-5 for g in tg.values()))
    item3 = True
    for f in ("bitident_det.json", "bitident_nondet.json", "bitident_actckpt.json", "bitident_actckpt_ipathg.json"):
        d = json.load(open(f"{V}/{f}"))
        item3 &= all(v["params_exact"] and v["optimizer_exact"] for r in d["steps"].values() for v in r.values())
        item3 &= all(v["n_exact"] == v["n_common"] for v in d["losses"].values())
    e = json.load(open(f"{V}/bitident_vs_existing_run.json"))
    item3 &= e["step0_params_exact"] and e["logged_losses_exact"] == e["logged_losses_common"] and e["batch_order"]["amix_vs_ipath_identical"]
    res["training_gate"] = {"item1_targets": item1, "item3_bitident": item3,
                            "item2_intervention": len(iv) == 5 and all(c["pass"] for c in res["intervention"].values()),
                            "S1_all_smoke_trainings_done": all(v["S1_done"] for v in res["smoke"].values())}
    res["training_gate"]["pass"] = all(res["training_gate"].values())
    res["evaluation_gate"] = {"smoke_rollouts_pass": all(v["pass"] for v in res["smoke"].values()),
                              "status": "required before ANY evaluation of matrix checkpoints"}
    pr = sorted(glob.glob(f"{V}/render_parity_egl_gpu3_run*.json"))
    res["evaluation_gate"]["gpu3_render_parity"] = {os.path.basename(f): json.load(open(f))["pass"] for f in pr}
    res["evaluation_gate"]["gpu3_parity_pass"] = len(pr) >= 3 and all(res["evaluation_gate"]["gpu3_render_parity"].values())
    pc = sorted(glob.glob(f"{V}/render_parity_egl_gpu3_ctxfix_run*.json"))
    res["evaluation_gate"]["gpu3_ctxfix_render_parity"] = {os.path.basename(f): json.load(open(f))["pass"] for f in pc}
    res["evaluation_gate"]["gpu3_ctxfix_parity_pass"] = len(pc) >= 3 and all(res["evaluation_gate"]["gpu3_ctxfix_render_parity"].values())
    res["evaluation_gate"]["ctxfix_note"] = ("evaluator frees the previous EGL context on every reset (ctxfix.py; robosuite "
                                             "hard-reset leak); parity re-verified 3x with the fix before test evaluation resumed")
    res["evaluation_gate"]["smoke_eval_dir"] = f"{R}/{EV}"
    res["evaluation_gate"]["pass"] = (res["evaluation_gate"]["smoke_rollouts_pass"] and res["evaluation_gate"]["gpu3_parity_pass"]
                                      and res["evaluation_gate"]["gpu3_ctxfix_parity_pass"])
    res["pass"] = res["training_gate"]["pass"] and res["evaluation_gate"]["pass"]
    json.dump(res, open(f"{V}/gate.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("training_gate", "evaluation_gate", "intervention")}, indent=1))
    ok = {"training": res["training_gate"]["pass"], "evaluation": res["evaluation_gate"]["pass"]}.get(which, res["pass"])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
