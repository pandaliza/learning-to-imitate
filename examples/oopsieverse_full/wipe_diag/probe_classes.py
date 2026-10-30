"""Raw segmentation class histogram (by env class name) per view, bank reset vs demo-state reset."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FULL = os.path.dirname(HERE)
for p in (os.path.join(FULL, "matrix"), os.path.join(FULL, "baseline"), FULL): sys.path.insert(0, p)
import rollout_eval as RE
import rollout_eval_matrix as REM
from wipe_common import demo_reset

def hist(env, obs):
    out = {}
    for cam in ["robot0_eye_in_hand", "robot0_agentview_right", "robot0_agentview_left"]:
        raw = np.asarray(obs[f"{cam}_segmentation_class"]).astype(int)
        u, c = np.unique(raw, return_counts=True)
        out[cam] = {f"{int(k)}:{env.seg_id_to_class.get(int(k), '?')}": int(n) for k, n in zip(u, c)}
    return out

def main():
    os.chdir(RE.OOPSIE)
    from ctxfix import install_ctx_free
    env = RE.make_env("wipe_countertop", use_camera_obs=False); install_ctx_free(env)
    res = {}
    for kind, r in [("bank", REM.dev_resets("wipe_countertop")[0]), ("demo", demo_reset("without", "demo_0"))]:
        obs, _ = RE.apply_reset(env, r)
        geom_classes = dict(getattr(env.model, "_geom_ids_to_classes", {}))
        cg = [g for g in range(env.sim.model.ngeom) if (env.sim.model.geom_id2name(g) or "").startswith("counter_main")]
        res[kind] = {"seg_id_to_class": {int(k): v for k, v in env.seg_id_to_class.items()}, "hist": hist(env, obs),
                     "ngeom": int(env.sim.model.ngeom), "n_geom_ids_to_classes": len(geom_classes),
                     "counter_main_geoms_sample": [(g, env.sim.model.geom_id2name(g), int(env.seg_lut[g]) if g < len(env.seg_lut) else None, geom_classes.get(g)) for g in cg[:8]]}
        print(kind, json.dumps(res[kind], indent=0)[:6000], flush=True)
    json.dump(res, open(os.path.join(sys.argv[1], "probe_classes.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
