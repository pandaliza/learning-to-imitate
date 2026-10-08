"""Bitwise check: generate_action_cached vs official CFMPolicy.generate_action on real dev samples (GPU 0)."""
import contextlib, json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.chdir("/edrive1/lizad/oopsieverse_full_study/src/oopsieverse"); sys.path.insert(0, os.getcwd())
from learning.models import CFMPolicy, PolicyConfig
from data import MatrixPlaybackDataset
from fast_native import generate_action_cached
res = {}
for run in sys.argv[1:]:
    m = json.load(open(f"{run}/meta.json")); pc = dict(m["policy_config"])
    pc["action_min"] = np.asarray(pc["action_min"], np.float32); pc["action_max"] = np.asarray(pc["action_max"], np.float32)
    pol = CFMPolicy(PolicyConfig(**pc)); pol.load_state_dict(torch.load(f"{run}/ckpt/step_30000.pth", map_location="cpu")["policy_state_dict"]); pol.cuda().eval()
    tc = m["trainer_config"]
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        ds = MatrixPlaybackDataset(data_path=m["dev_hdf5"], frame_stack=2, action_chunk_size=8, seg_img_size=(128, 128), normalize_action=True,
                                   objects_of_interest=tc["objects_of_interest"], obs_keys=tc["obs_keys"])
    g = torch.Generator().manual_seed(0); maxd = 0.0; exact = True; n = 0
    for B in (1, 3, 9, 16):
        idx = torch.randint(len(ds), (B,), generator=g).tolist(); bt = [ds[i] for i in idx]
        seg = {k: torch.stack([b["obs"]["extero"][k] for b in bt]).cuda() for k in bt[0]["obs"]["extero"]}
        st = torch.stack([b["obs"]["proprio"] for b in bt]).cuda()
        x0 = torch.randn(B, 8, pol.config.action_dim, generator=g).cuda()
        a = pol.generate_action(seg, st, init_noise=x0); b = generate_action_cached(pol, seg, st, x0)
        exact &= bool(torch.equal(a, b)); maxd = max(maxd, float((a - b).abs().max())); n += B
    res[run] = {"n_samples": n, "batch_sizes": [1, 3, 9, 16], "bitwise_equal": exact, "max_abs_diff": maxd}
print(json.dumps(res, indent=1))
json.dump(res, open("/scr/lizad/personal/learning-to-imitate/outputs/oopsieverse_full_study/matrix/verification/verify_fast_native.json", "w"), indent=1)
