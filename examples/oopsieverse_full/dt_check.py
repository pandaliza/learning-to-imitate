"""Controlled check of the mechanical-damage dt mechanism (spec section 6). Released env is NOT patched.

Part 1 (simulator, pick_egg): drop the egg from controlled heights with the robot holding still.
  Per control step we log the released evaluator's impact force (dt = 0.005 s, hard-coded at
  damagesim/robosuite/evaluators/mechanical.py:113-116), the same finite difference at the
  control-rate dt (0.05 s; computed alongside, not substituted into the env), the quasistatic force,
  the released damage and the damage the formula would give at dt = 0.05. Substeps are instrumented
  (wrapping sim.step2 on this test instance only) to find the substep of first egg-counter contact,
  the true per-substep peak normal force, and whether the evaluator registers the impact in the
  same control step as contact onset (one-step event offset check).
  We also measure the cvel lag: cvel read after env.step vs cvel after a fresh forward() of the
  same post-step state.

Part 2 (offline, exact): recompute every stored step's damage at dt = 0.05 from the stored
  per-step impact/qs forces. The impact term scales exactly with 1/dt; qs is unaffected by the dt
  magnitude (only the acceleration direction enters, core/evaluators/mechanical.py:172-199).
"""

import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, TASKS, CONDITIONS, teleop_path, demo_sort_key, object_final_health  # noqa: E402
from replay import get_env  # noqa: E402

DT_RELEASED = 0.005
DT_CONTROL = 0.05
EGG = dict(thr=15.0, scale=1.0, s_imp=10.0, s_qs=1.0)  # damage_params.py egg entry; verified against stored data


def egg_damage(imp, qs, p=EGG):
    return np.minimum(100.0, np.maximum(0.0, p["s_imp"] * imp + p["s_qs"] * qs - p["thr"]) * p["scale"])


def part1(heights_cm=(0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30), lateral_offsets=(0.0, 0.0007, 0.0013)):
    env = get_env("pick_egg")
    res = []
    for off in lateral_offsets:  # small offsets shift the contact phase within the control step
        for h in heights_cm:
            env.reset()
            sim = env.sim  # reset may rebuild the sim
            egg = env.objects["egg"]
            bid = env.obj_body_id["egg"]
            mass = float(sim.model.body_mass[bid])
            jn = egg.joints[0]
            qa = sim.model.get_joint_qpos_addr(jn)
            va = sim.model.get_joint_qvel_addr(jn)
            z = np.zeros(env.action_dim)
            for _ in range(10):  # settle with zero action
                env.step(z)
            q = sim.data.qpos.copy()
            q[qa[0] + 2] += h / 100.0
            q[qa[0]] += off * h  # tiny lateral shift varies the free-fall time slightly
            sim.data.qpos[:] = q
            sim.data.qvel[va[0]:va[1]] = 0.0
            sim.forward()
            ev = [e for e in egg.damage_evaluators if e.name == "mechanical"][0]
            ev.prev_linear_velocities["egg_main"] = ev._get_part_linear_velocity("egg_main")
            h0 = float(egg.link_healths["egg_main"])
            egg_geoms = {g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == bid}
            sub = []
            orig_step2 = sim.step2

            def step2_logged():
                orig_step2()
                fsum = 0.0
                for i in range(sim.data.ncon):
                    c = sim.data.contact[i]
                    if (c.geom1 in egg_geoms or c.geom2 in egg_geoms) and c.efc_address >= 0:
                        fsum += abs(sim.data.efc_force[c.efc_address])
                sub.append(fsum)

            sim.step2 = step2_logged
            steps = []
            try:
                for k in range(int(np.ceil(np.sqrt(2 * h / 100.0 / 9.81) / 0.05)) + 8):
                    sub.clear()
                    obs, _, _, info = env.step(z)
                    m = info["damage_info"]["egg"]["egg_main"]["mechanical"]
                    v_lag = sim.data.cvel[bid][3:6].copy()
                    sim.forward()
                    v_true = sim.data.cvel[bid][3:6].copy()
                    steps.append(dict(k=k, impact_released=m["impact_forces"], impact_control_dt=m["impact_forces"] * DT_RELEASED / DT_CONTROL,
                                      qs=m["filtered_qs_forces"], damage_released=m["damage"],
                                      damage_control_dt=float(egg_damage(m["impact_forces"] * DT_RELEASED / DT_CONTROL, m["filtered_qs_forces"])),
                                      health=float(obs["health"][0]), sub_peak_force=max(sub) if sub else 0.0,
                                      first_contact_substep=next((j for j, fv in enumerate(sub) if fv > 1e-6), None),
                                      cvel_lag_norm=float(np.linalg.norm(v_true - v_lag))))
            finally:
                sim.step2 = orig_step2
            # contact onset step (first step whose substeps saw egg contact after release)
            onset = next((s["k"] for s in steps if s["first_contact_substep"] is not None), None)
            imp_step = int(np.argmax([s["impact_released"] for s in steps]))
            v_imp = float(np.sqrt(2 * 9.81 * h / 100.0))
            res.append(dict(height_cm=h, lateral_offset=off, egg_mass_kg=mass, impact_speed_mps=v_imp,
                            health_before=h0, health_after=steps[-1]["health"],
                            loss_released=h0 - steps[-1]["health"],
                            loss_control_dt=float(sum(s["damage_control_dt"] for s in steps)),
                            max_impact_released_N=max(s["impact_released"] for s in steps),
                            max_impact_control_dt_N=max(s["impact_control_dt"] for s in steps),
                            momentum_over_control_dt_N=mass * v_imp / DT_CONTROL,
                            max_qs_N=max(s["qs"] for s in steps),
                            true_peak_substep_contact_N=max(s["sub_peak_force"] for s in steps),
                            contact_onset_step=onset,
                            onset_substep=steps[onset]["first_contact_substep"] if onset is not None else None,
                            peak_impact_step=imp_step,
                            impact_registered_one_step_late=(onset is not None and imp_step == onset + 1),
                            max_cvel_lag_mps=max(s["cvel_lag_norm"] for s in steps), steps=steps))
            print(f"h={h}cm off={off} loss_rel={res[-1]['loss_released']:.2f} loss_ctl={res[-1]['loss_control_dt']:.2f} "
                  f"Fimp_rel={res[-1]['max_impact_released_N']:.2f} Fpeak_sub={res[-1]['true_peak_substep_contact_N']:.1f} "
                  f"onset={onset}/{res[-1]['onset_substep']} peak_step={imp_step}", flush=True)
    return res


def part2():
    rows = []
    for task in TASKS:
        for cond in CONDITIONS:
            with h5py.File(teleop_path(task, cond), "r") as f:
                for d in sorted(f["data"], key=demo_sort_key):
                    g = f["data"][d]
                    names = [n.decode() if isinstance(n, bytes) else str(n) for n in g.attrs["health_list_link_names"]]
                    di = [json.loads(x) for x in g["info/damage_info"][:]]
                    H = g["obs/health"][:].astype(np.float64)
                    # recompute per-link per-step damage both ways with the stored per-link parameters:
                    # stored damage is exactly the released formula; infer the per-link (s_imp) from where impact enters.
                    h_rel = np.full(len(names), 100.0)
                    h_ctl = np.full(len(names), 100.0)
                    max_abs_formula_err = 0.0
                    for t, dd in enumerate(di):
                        for k, n in enumerate(names):
                            o, l = n.split("@", 1)
                            m = dd.get(o, {}).get(l, {}).get("mechanical")
                            if m is None:
                                continue
                            if o == "egg":
                                p = EGG
                            else:  # robot links and sponge: impact sensitivity 0 (damage_params.py), dt cannot matter
                                p = None
                            if p is not None:
                                dr = float(egg_damage(m["impact_forces"], m["filtered_qs_forces"], p))
                                max_abs_formula_err = max(max_abs_formula_err, abs(dr - m["damage"]))
                                dc = float(egg_damage(m["impact_forces"] * DT_RELEASED / DT_CONTROL, m["filtered_qs_forces"], p))
                            else:
                                dr = dc = m["damage"]
                            h_rel[k] = max(0.0, h_rel[k] - dr)
                            h_ctl[k] = max(0.0, h_ctl[k] - dc)
                    fr = object_final_health(h_rel, names)
                    fc = object_final_health(h_ctl, names)
                    fs = object_final_health(H[-1], names)
                    rows.append(dict(task=task, cond=cond, demo=d,
                                     stored_final=fs, recomputed_released=fr, control_dt=fc,
                                     recompute_vs_stored_max_abs=max(abs(fr[o] - fs[o]) for o in fs),
                                     formula_max_abs_err=max_abs_formula_err))
    return rows


def labels(final, tol=1e-4):
    vals = list(final.values())
    return dict(paper=all(v > 95.0 for v in vals), code=float(np.mean(vals)) >= 95.0, strict=all(v >= 100.0 - tol for v in vals))


def main():
    out = {"dt_released": DT_RELEASED, "dt_control": DT_CONTROL, "egg_params": EGG}
    p2 = part2()
    summ = {}
    for task in TASKS:
        for cond in CONDITIONS + ["all"]:
            rs = [r for r in p2 if r["task"] == task and (cond == "all" or r["cond"] == cond)]
            s = {}
            for rule in ["paper", "code", "strict"]:
                a = [labels(r["recomputed_released"])[rule] for r in rs]
                b = [labels(r["control_dt"])[rule] for r in rs]
                s[rule] = dict(safe_released=int(sum(a)), safe_control_dt=int(sum(b)),
                               flips_damaging_to_safe=int(sum((not x) and y for x, y in zip(a, b))),
                               flips_safe_to_damaging=int(sum(x and (not y) for x, y in zip(a, b))), n=len(rs))
            summ[f"{task}/{cond}"] = s
    out["part2_label_summary_health_only"] = summ
    out["part2_max_recompute_vs_stored_abs"] = max(r["recompute_vs_stored_max_abs"] for r in p2)
    out["part2_max_formula_err"] = max(r["formula_max_abs_err"] for r in p2)
    out["part2_rows"] = p2
    print(json.dumps(summ, indent=1))
    out["part1_drops"] = part1()
    os.makedirs(os.path.join(OUT, "dt_check"), exist_ok=True)
    with open(os.path.join(OUT, "dt_check", "dt_check.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
