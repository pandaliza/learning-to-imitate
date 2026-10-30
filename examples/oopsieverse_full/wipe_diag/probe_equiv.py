"""Is the 'alias' LUT (env class 'Counter' -> vocab 'DamageableCounter') exactly the segmentation the policy saw in
training? For K eval_dev bank resets: (1) plain bank reset (frozen evaluator path) -> policy seg with frozen LUT and
with alias LUT; (2) re-load the SAME reset through the playback path (set_ep_meta(get_ep_meta()) -> reset ->
reset_from_xml_string(model xml) -> set_state(state)), as official playback does for training data -> policy seg
with the frozen LUT. Report pixel agreement per view."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.dirname(HERE)
for p in (os.path.join(FULL, "matrix"), os.path.join(FULL, "baseline"), FULL, HERE):
    sys.path.insert(0, p)
import rollout_eval as RE  # noqa: E402
import rollout_eval_matrix as REM  # noqa: E402
from diag_rollout import AliasPolicy  # noqa: E402
from wipe_common import VOCAB_RUN  # noqa: E402


def segs(pol, obs, lut):
    s, _ = pol.frame(obs, lut)
    return {k.split("/")[1]: v for k, v in s.items()}


def main():
    os.chdir(RE.OOPSIE)
    from ctxfix import install_ctx_free
    env = RE.make_env("wipe_countertop", use_camera_obs=False)
    install_ctx_free(env)
    pol = RE.Policy(VOCAB_RUN, ckpt="step_30000.pth", load_model=False)
    rows = []
    for r in REM.dev_resets("wipe_countertop")[: int(sys.argv[2])]:
        obs, _ = RE.apply_reset(env, r)
        frozen = segs(pol, obs, RE.Policy.seg_lut(pol, env.seg_id_to_class))
        alias = segs(pol, obs, AliasPolicy.seg_lut(pol, env.seg_id_to_class))
        ep_meta, xml = env.get_ep_meta(), env.sim.model.get_xml()
        state = env.sim.get_state().flatten().copy()
        names_bank = sorted(set(env.seg_id_to_class.values()))
        obs2, _ = RE.apply_reset(env, {"seed": r["seed"], "ep_meta": ep_meta, "model_xml": xml, "state": state.tolist()})
        pb = segs(pol, obs2, RE.Policy.seg_lut(pol, env.seg_id_to_class))
        st_err = float(np.abs(env.sim.get_state().flatten() - state).max())
        row = {"id": r["id"], "state_maxabs_diff": st_err,
               "fixture_cls_in_ep_meta": sorted({v["cls"] for v in ep_meta.get("fixtures", {}).values()})[:4],
               "bank_class_names_sample": names_bank[:6], "playback_class_names_sample": sorted(set(env.seg_id_to_class.values()))[:6]}
        for cam in pb:
            row[cam] = {"frozen_vs_playback_agree": float((frozen[cam] == pb[cam]).mean()),
                        "alias_vs_playback_agree": float((alias[cam] == pb[cam]).mean()),
                        "counter_px_frozen": int((frozen[cam] == 1).sum()), "counter_px_playback": int((pb[cam] == 1).sum())}
        rows.append(row)
        print(json.dumps(row), flush=True)
    json.dump(rows, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
