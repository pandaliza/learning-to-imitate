"""EGL render-context leak fix for evaluation (wrapper only; robosuite and official code unchanged).

robosuite's hard reset (base.py reset) only calls _destroy_sim() when renderer == "mujoco". Our evaluator env has
has_renderer=False, so each reset builds a new MjSim + MjRenderContextOffscreen while the previous sim and its EGL
context stay alive (measured: +1.08 GB of GPU-3 memory per episode per worker). install_ctx_free(env) wraps
env.reset to free the previous offscreen context (mjrContext + EGL context) before the reset creates the new one.
Rendering of the new context is unchanged (re-verified by the exact pixel-parity check, render_parity.py ctxfix).
"""
import gc


def free_offscreen(env):
    sim = getattr(env, "sim", None)
    ctx = getattr(sim, "_render_context_offscreen", None) if sim is not None else None
    if ctx is None:
        return False
    for f in (lambda: ctx.con.free(), lambda: ctx.gl_ctx.free()):
        try:
            f()
        except Exception:
            pass
    sim._render_context_offscreen = None
    return True


def install_ctx_free(env):
    orig = env.reset

    def reset(*a, **k):
        free_offscreen(env)
        gc.collect()
        return orig(*a, **k)
    env.reset = reset
    return env
