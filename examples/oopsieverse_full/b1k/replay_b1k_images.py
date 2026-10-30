"""Image-producing official B1K playback for policy training data (B1K diagnostic arm, protocol Amendment 1).

Runs the SAME instrumented official playback as replay_b1k.py (imported, not edited; frozen shim in
replay_b1k_shim.diff), with:
  * --with_images (official low_resolution: 256x256 rgb + seg_instance on eef + external_sensor0/1, i.e. the
    official PlaybackDataset default obs keys); external cameras are NOT dropped.
  * extra read-only instrumentation for the obs/action alignment audit: every post-step state P_i (state after
    official step i = load stored s[i+1], env.step(a[i])) is compared with stored s[i+1] AND s[i+2]
    (robot joint positions, object positions). Playback obs[k] is P_{k+4} (initial frame recorded at i=4,
    init_skip_steps=4) and playback action[k] = teleop a[k+5].
  * optional --audit_pass_a: the state forced before step i is s[i] instead of s[i+1] (official uses s[i+1]),
    i.e. "restore s[i], step a[i]"; P_i is then compared with s[i+1]. Audit-only; output HDF5 is discarded.
Writes metrics JSON (per-episode timing + both divergences) to --metrics_json.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import replay_b1k as R  # noqa: E402  (chdir + official imports happen here)
import torch as th  # noqa: E402

_orig_div = R._divergence


def _robot_joint_err(post_states, stored_states, stored_sizes, offset):
    """max |joint_pos| error of franka0 between P_i and stored s[i+offset]; also eef-link-free object pos err."""
    import omnigibson as og
    out = []
    for i, ps in enumerate(post_states):
        j = i + offset
        if j >= len(stored_states):
            break
        ref, _ = R._shim_fix(stored_states[j], int(stored_sizes[j]))
        if ps.numel() != ref.numel():
            out.append([i, None, None])
            continue
        a = R._flatten(og.sim.deserialize(ps)[0])
        b = R._flatten(og.sim.deserialize(ref)[0])
        rj = [k for k in a if "franka0" in k and k.endswith("joint_pos") and k in b]
        jerr = max(float((a[k] - b[k]).abs().max()) for k in rj) if rj else None
        perr = [float(th.linalg.norm(a[k] - b[k])) for k in a if k in b and k.endswith("/pos") and a[k].numel() == 3
                and "franka0" not in k]
        out.append([i, jerr, max(perr) if perr else None])
    return out


def _divergence(post_states, stored_states, stored_sizes):
    d = _orig_div(post_states, stored_states, stored_sizes)
    if os.environ.get("B1K_ALIGN_INSTR", "0") != "1":  # alignment audit finished 2026-10-23; cost-only switch
        return d
    d["align_vs_s_i+1"] = _robot_joint_err(post_states, stored_states, stored_sizes, 1)
    d["align_vs_s_i+2"] = _robot_joint_err(post_states, stored_states, stored_sizes, 2)
    # reference: one recorded control step of motion, stored s[i+1] vs stored s[i+2]
    ref = [R._shim_fix(stored_states[i + 1], int(stored_sizes[i + 1]))[0] for i in range(len(post_states))
           if i + 1 < len(stored_states)]
    d["stored_step_s_i+1_vs_s_i+2"] = _robot_joint_err(ref, stored_states, stored_sizes, 2)
    return d


R._divergence = _divergence


def _enable_pass_a():
    """Audit only: force s[i] (previous distinct stored state) instead of s[i+1] before each playback step."""
    from damagesim.omnigibson.damageable_env import OGDamageableDataPlaybackWrapper as W
    inner = W._load_state_with_size_fallback
    st = {"cur": None, "prev": None, "n": 0}

    def _load(self, state, saved_size):
        if st["cur"] is None or state.data_ptr() != st["cur"][0].data_ptr():
            st["prev"] = st["cur"]
            st["cur"] = (state, saved_size)
        use = st["prev"] if st["prev"] is not None else st["cur"]
        st["n"] += 1
        return inner(self, use[0], use[1])

    def reset_ptr():
        st.update(cur=None, prev=None)

    W._load_state_with_size_fallback = _load
    return reset_ptr


if __name__ == "__main__":
    if "--audit_pass_a" in sys.argv:
        sys.argv.remove("--audit_pass_a")
        # the shim (if enabled) replaces W._load_state_with_size_fallback inside _attach; wrap after it.
        _orig_attach = R._attach_instrumented

        def _attach(env, task_cfg, task_mod):
            _orig_attach(env, task_cfg, task_mod)
            reset_ptr = _enable_pass_a()
            R.REC["audit_pass_a"] = True
            orig_ep = env.playback_episode

            def ep(episode_id, **kw):
                reset_ptr()
                return orig_ep(episode_id=episode_id, **kw)
            env.playback_episode = ep
        R.pb.attach_task_playback_hooks = _attach
    R.main()
