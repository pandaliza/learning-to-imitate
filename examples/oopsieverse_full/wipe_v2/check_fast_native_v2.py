"""fast_native.generate_action_cached == official generate_action for W1/W2 (random init, CPU, fixed noise)."""
import json, sys, os, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "matrix"))
import models_v2 as M2
from fast_native import generate_action_cached
meta = json.load(open("/edrive1/lizad/oopsieverse_full_study/runs/matrix/amix_wipe_countertop_v1_s0/meta.json"))
out = {}
for arm in ("w1", "w2"):
    torch.manual_seed(0); pol = M2.build_policy(arm, M2.policy_config_from_meta(meta), 0).eval()
    seg = {f"v{i}": torch.randint(0, 5, (3, 2, 128, 128)) for i in range(3)}; st = torch.randn(3, 23); x0 = torch.randn(3, 8, 12)
    a = pol.generate_action(seg, st, init_noise=x0); b = generate_action_cached(pol, seg, st, x0)
    out[arm] = {"bitwise_equal": bool(torch.equal(a, b)), "maxabs": float((a - b).abs().max())}
print(out)
