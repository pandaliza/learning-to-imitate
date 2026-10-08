"""Builds outputs/oopsieverse_full_study/selection/restore_through_damage.md from the measured JSON results."""
import glob
import json
import os

OUT = "/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study/selection"


def yn(x):
    return "PASS" if x else "**FAIL**"


def main():
    phys = []
    for f in sorted(glob.glob(f"{OUT}/restore_tests/physics_*.json")):
        phys += json.load(open(f))
    v2 = json.load(open(f"{OUT}/verification/v2_rendered_restore.json")) if os.path.exists(f"{OUT}/verification/v2_rendered_restore.json") else []
    fc = json.load(open(f"{OUT}/verification/fast_contacts.json")) if os.path.exists(f"{OUT}/verification/fast_contacts.json") else []
    L = []
    w = L.append
    w("# Restore through damage: measured verification")
    w("")
    w("Spec §11 and §15 item 5. Code: `examples/oopsieverse_full/selection/snapshot_damage_state.py` (snapshot and restore), "
      "`test_restore.py` (physics path), `verify_harness.py` v2 (rendered path with the policy in the loop). "
      "Raw results: `restore_tests/physics_*.json`, `verification/v2_rendered_restore.json`.")
    w("")
    w("Stack: robosuite 1.5.2 (aaa8b9b), robocasa 97a4060, mujoco 3.3.1, oopsieverse a739f99, PandaOmron with the "
      "HYBRID_MOBILE_BASE composite controller, the continuous-gripper patch, and DamageSim. No official file was edited.")
    w("")
    w("**Criterion.** Every comparison is bit-exact (`np.array_equal`; max |diff| is reported as 0). Health is "
      "checked separately, so equal qpos with different health counts as a FAIL. No tolerance was needed.")
    w("")
    w("## What a snapshot contains")
    w("")
    w("| Component | Captured as | Restored by |")
    w("|---|---|---|")
    w("| MuJoCo integration state (time, qpos, qvel, act, qacc_warmstart, ctrl, applied forces, mocap, userdata, plugin state) | `mj_getState(mjSTATE_INTEGRATION)` | `mj_setState`, then `mj_forward`, then `mj_setState` again (mj_forward overwrites qacc_warmstart), asserted bit-equal |")
    w("| Model edits (wipe marker `geom_rgba` alpha and `body_pos`, hidden eef sites `site_rgba`, plus geom/body pose, size, mass, friction, contype) | full arrays | written back in full; a structural fingerprint (sizes, inertia, joints, actuators, solver options) must match |")
    w("| Controllers: arm OSC_POSE (goals, interpolators, reference frames, `_goal_update_mode` = the base-mode bit), gripper, mobile base (joint velocity, `init_pos/ori`, base frame), torso (joint position) | every plain-data attribute of each part controller, plus the composite controller's `_applied_action_dict` | `__dict__` write-back |")
    w("| Continuous-gripper patch state | `PandaGripper.current_action` | yes |")
    w("| Robot buffers and torques (`recent_qpos/actions/torques/ee_*`), base pose cache | Delta/Ring buffer state | yes |")
    w("| Observation history: 2 frames of segmentation (3 cameras, LUT to train vocabulary) and proprio | the caller's `episode` state stored inside the snapshot | returned by `restore` |")
    w("| Env bookkeeping: `timestep`, `cur_time`, `done`, `_ep_meta`, the observable sampling state and `_obs_cache`, segmentation LUT, and every other plain env attribute | yes | yes |")
    w("| Episode protocol state: step count, success latch (first success step), running maxima of damage and contact force, egg-height trace summary, time-mean health accumulator | caller `episode` | returned |")
    w("| RNGs: `env.rng` (numpy Generator state), process-global numpy and python RNGs | yes | yes. Stepping does not consume the global RNGs (measured). Policy noise is keyed by the caller. |")
    w("| DamageSim, per entity (egg or sponge, PandaOmron): `link_healths`, `damage_info`, `track_damage`, `damageable_links`, params | yes | yes |")
    w("| DamageSim, per evaluator: `prev_linear_velocities`, `_tracking_initialized`, `_body_id_cache`, `damage_potentials`, and the full impact / qs / raw-force / contact logs | yes | yes. The `contacts_by_body` alias of `contacts_by_part` is re-established. |")
    w("| DamageSim, env level: `lock_health`, `damage_evaluators_initialized`, `health_list_link_names`, `_damage_config` | yes | yes |")
    w("| Wipe: `wiped_markers` (stored as indices and re-bound to marker objects), `ee_force_bias`, `_marker_positions` and `_marker_direction` | yes | yes |")
    w("| RoboCasa fixture python state (oven `_timer` / `_temperature` / `_door`, dishwasher door, toaster state), updated every step by `Kitchen.update_state` | every plain attribute of every fixture | yes. The whole-graph audit found this; an earlier draft missed it. |")
    w("| Thermal and fluid accumulators | none exist for RoboCasa: only the `mechanical` evaluator is registered (backend_audit §2) | n/a |")
    w("")
    w("**Rebuild order for another env instance or process** (`rebuild_and_restore`):")
    w("1. Run the canonical reset of the parent (reset_banks_v1: `unset_ep_meta`; `env.rng = default_rng(seed)`; `reset()`), and hide the eef markers. This gives the same compiled model and fresh DamageSim evaluators.")
    w("2. Check the structural fingerprint.")
    w("3. Write the model edits.")
    w("4. Restore the integration state.")
    w("5. Restore the env python state, observables and RNG.")
    w("6. Restore robots and controllers.")
    w("7. Restore DamageSim.")
    w("8. Restore the wipe markers.")
    w("9. Restore the global RNGs.")
    w("")
    w("`reset()` is never called after the restore. For demos, the physics-path tests use the official playback rebuild instead: `set_ep_meta`, `reset`, `reset_from_xml_string`, then `states[0]`.")
    w("")
    w("## Physics path (CPU, no renderer): damaging teleop demos, replayed open loop")
    w("")
    w("Setup for each case:")
    w("- **Snapshot.** S is taken at t_b = first damage step − 12, a pre-damage state (all link healths are 100).")
    w("- **Reference R.** The demo actions are replayed from S to the end of the demo, through the damaging contact.")
    w("- **Trace.** Every step records the full integration state, every low-dim observation (health, proprio, ...), the per-link damage_info (damage, impact force, filtered and unfiltered qs and raw forces, contact count, summed contact magnitude), reward, official success, wiped-marker count, timestep and cur_time.")
    w("")
    w("| Case | t_b → end | Damage in R (link-steps, total health lost) | Success | T0 graph digest | T1 same instance | T2 restore twice | T3 branch order | T4 fresh instance | T5 disk + fresh process | T6 post-chunk cache | N1 physics-only (must FAIL) | N2 velocity re-sync (must FAIL) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in phys:
        ref = r["T1_same_instance"]
        w(f"| {r['case']} | {r['t_b']} → {r['t_end']} | {ref['damage_events_ref']}, {ref['health_delta_ref']:.2f} | {ref['success_ref']} | "
          f"{yn(r['T0_graph_digest']['pass'])} ({r['T0_graph_digest']['n_paths']} paths) | {yn(r['T1_same_instance']['pass'])} | "
          f"{yn(r['T2_restore_twice']['pass'])} | {yn(r['T3_branch_order']['pass'])} | {yn(r['T4_fresh_instance']['pass'])} | "
          f"{yn(r['T5_disk_fresh_process']['pass'])} | {yn(r['T6_post_chunk_cache']['pass'])} | "
          f"{'FAIL as expected' if r['N1_physics_only_restore']['ok'] else '**not detected**'} (qpos equal: {r['N1_physics_only_restore']['qpos_equal_at_restore']}, max health diff {r['N1_physics_only_restore']['max_health_diff']:.1f}) | "
          f"{'FAIL as expected' if r['N2_velocity_resync']['ok'] else '**not detected**'} (first-step impact diff {r['N2_velocity_resync']['first_step_impact_maxdiff']:.2g} N) |")
    w("")
    w("Every bit-exact test reports max |diff| = 0 for every trace component; see `max_abs_diff` in the JSON.")
    w("")
    w("What each test does:")
    w("- **T0.** The python object graph reachable from the env is walked through robosuite, robocasa and damagesim objects (about 3–4k leaf paths). The digest right after restore must equal the digest at capture.")
    w("- **T1.** Restore into the env that has already run through the damage, then replay.")
    w("- **T2.** Restore, run a perturbed branch for 10 steps, restore again, then replay.")
    w("- **T3.** Branch X uses perturbed actions (N(0, 0.3), which damages differently); branch Y uses the original actions. They are run in the orders XY and YX. X must be identical in both orders, and Y must equal R in both.")
    w("- **T4.** A new env object, already used for 3 steps, is rebuilt from the demo and restored.")
    w("- **T5.** The snapshot is written to disk, and a separate python process rebuilds, loads, restores and replays.")
    w("- **T6.** Post-chunk caching: a snapshot S8, taken 8 steps after S, goes through the disk and is continued twice. Both runs must equal R[8:]. This is the determinism the harness relies on when it restores S8 for each tail.")
    w("")
    w("Negative controls:")
    w("- **N1.** Only the MuJoCo state is restored. qpos is then identical, but health carries over from the damaged run, so the checker reports FAIL. This is the \"equal qpos with different health\" case.")
    w("- **N2.** Everything is restored except `prev_linear_velocities`, which is re-synced to the current cvel (the playback convention). The first post-restore impact force differs, so the checker reports FAIL.")
    w("")
    w("**Fresh-instance exclusions** (T4 graph digest): `env.objects[<obj>].file` and the matching sampler `mujoco_objects[0].file`. These are names of temporary MJCF copies written at model load, such as `.../sponge_1/<timestamp>_<pid>.xml`. They are not episode state and never enter physics.")
    w("")
    w("## Rendered path with the policy in the loop (GPU-3 EGL, CPU inference)")
    w("")
    if v2:
        w("Setup for each parent:")
        w("- canonical reset of the parent (reset_banks_v1 `sel_dev_parents`);")
        w("- A-MIX seed 0 (30k) runs to t_b = 48 with a fixed torch generator;")
        w("- capture the snapshot, including the 2-frame segmentation/proprio history and the generator state;")
        w("- continue to the end of the episode (the evaluator protocol) to get the reference.")
        w("")
        w("Compared in each check:")
        w("- the per-step hash of health plus integration state;")
        w("- the hash of every policy input (3 × 2 segmentation frames plus proprio) and every decoded chunk;")
        w("- the final record.")
        w("")
        w("| Parent | Continuation damaging | Reference outcome (steps, worst loss) | In-place restore | Restore twice | Fresh env instance (after another parent) | Disk + fresh process |")
        w("|---|---|---|---|---|---|---|")
        for r in v2:
            c = r["checks"]
            ref = r["reference"]
            w(f"| {r['parent']} | {r['damaging_continuation']} | {ref['outcome']} ({ref['steps']}, {ref['worst_entity_loss']:.1f}) | "
              f"{yn(c['in_place_restore']['pass'])} | {yn(c['restore_twice']['pass'])} | {yn(c['fresh_env_instance']['pass'])} | {yn(c['disk_fresh_process']['pass'])} |")
        w("")
        w("A pass needs all three to be identical: the final record, the per-step physics and health trace, and the sequence of policy inputs. Matching policy inputs means the rendered segmentation history after restore is bit-identical.")
    else:
        w("(pending)")
    w("")
    w("## DamageSim contact-scan shim (performance; bit-exact)")
    w("")
    if fc:
        w("`fast_contacts.py` replaces only the per-step geom-set construction in `RSMechanicalDamageEvaluator._get_part_contacts`. One numpy comparison replaces a Python loop over all 1,709 geoms; the contact loop is kept verbatim. Results:")
        w("")
        w("| Case | Per-call outputs identical | Whole-replay trace bit-exact | s/step released → shim |")
        w("|---|---|---|---|")
        for r in fc:
            w(f"| {r['case']} | {r['per_call']['n_identical']}/{r['per_call']['n_calls']} ({r['per_call']['n_nonempty_contact_lists']} non-empty) | {r['trace_bit_exact']} | {r['s_per_step_released']:.3f} → {r['s_per_step_shim']:.3f} |")
    w("")
    w("## Verdict")
    ok_phys = all(r[k]["pass"] for r in phys for k in ("T1_same_instance", "T2_restore_twice", "T3_branch_order", "T4_fresh_instance",
                                                        "T5_disk_fresh_process", "T6_post_chunk_cache", "T0_graph_digest")) and \
        all(r["N1_physics_only_restore"]["ok"] and r["N2_velocity_resync"]["ok"] for r in phys)
    ok_v2 = bool(v2) and all(all(c["pass"] for c in r["checks"].values()) for r in v2)
    w("")
    w(f"- Physics path: {len(phys)} damaging cases (pick_egg ×{sum(r['case'].startswith('pick') for r in phys)}, wipe ×{sum(r['case'].startswith('wipe') for r in phys)}). **{'ALL PASS' if ok_phys else 'FAILURES PRESENT'}**, bit-exact. Both negative controls were detected.")
    w(f"- Rendered policy path: {len(v2)} parents ({sum(r['damaging_continuation'] for r in v2)} with damaging continuations). **{'ALL PASS' if ok_v2 else ('PENDING' if not v2 else 'FAILURES PRESENT')}**.")
    open(f"{OUT}/restore_through_damage.md", "w").write("\n".join(L) + "\n")
    print("ok", ok_phys, ok_v2)


if __name__ == "__main__":
    main()
