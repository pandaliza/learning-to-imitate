"""CPU forward/backward shape test for W0-W3 + check that shared modules start from W0's init and the global RNG
stream after construction equals W0's."""
import json, sys, numpy as np, torch
import models_v2 as M2
meta = json.load(open("/edrive1/lizad/oopsieverse_full_study/runs/matrix/amix_wipe_countertop_v1_s0/meta.json"))
pcfg = M2.policy_config_from_meta(meta)
out = {}
B = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ref_sd, ref_next = None, None
for arm in M2.ARMS_V2:
    torch.manual_seed(0)
    pol = M2.build_policy(arm, pcfg, 0)
    nxt = torch.rand(3).tolist()
    sd = pol.state_dict()
    if arm == "w0":
        ref_sd, ref_next = sd, nxt
    shared = [k for k in sd if k in ref_sd and sd[k].shape == ref_sd[k].shape and not k.startswith("pos_encoding") and not (arm != "w0" and k.startswith("seg_encoder.projection"))]
    same = all(torch.equal(sd[k], ref_sd[k]) for k in shared)
    torch.manual_seed(1)
    seg = {f"v{i}": torch.randint(0, 5, (B, 2, 128, 128)) for i in range(3)}
    state = torch.randn(B, 23 + (5 if arm == "w3" else 0))
    act = torch.randn(B, 8, 12)
    obs_tok = pol.obs_tokens(seg, state[:, :23]) if arm == "w3" else pol.encode_obs(seg, state)
    loss = pol.compute_loss(seg, state, act)
    loss.backward()
    nograd = [n for n, p in pol.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    pol.eval()
    a = pol.generate_action(seg, state[:, :23] if arm != "w3" else state)
    out[arm] = {"params": sum(p.numel() for p in pol.parameters()), "obs_tokens": list(obs_tok.shape),
                "num_obs_tokens": int(pol.num_obs_tokens), "loss": float(loss), "action_shape": list(a.shape),
                "params_without_grad": nograd, "n_shared_tensors": len(shared), "shared_init_equal_w0": same,
                "global_rng_after_init_equal_w0": nxt == ref_next}
print(json.dumps(out, indent=1))
json.dump(out, open("/edrive1/lizad/oopsieverse_full_study/reports/wipe_v2/verification/shape_test_cpu.json", "w"), indent=1)
