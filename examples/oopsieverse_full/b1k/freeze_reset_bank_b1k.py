"""Assemble + freeze reset_bank_b1k_v1.json from the shard outputs (b1k_diag_protocol.md §4). Refuses to overwrite."""
import glob
import hashlib
import json
import os

SH = "/edrive1/lizad/oopsieverse_full_study/runs/b1k_diag/reset_bank/shards"
OUT = "/edrive1/lizad/oopsieverse_full_study/reports/b1k/reset_bank_b1k_v1.json"
POOLS = {"test": (100000, 50), "dev": (200000, 30)}


def main():
    assert not os.path.exists(OUT), "frozen bank exists"
    bank = {"status": "FROZEN b1k reset bank v1 (DIAGNOSTIC); built before any B1K policy evaluation",
            "protocol": "b1k_diag_protocol.md section 4", "tasks": {}}
    for task in ("add_firewood", "shelve_item"):
        t = {"pools": {}, "determinism": [], "shards": []}
        for f in sorted(glob.glob(f"{SH}/SHARD_{task}_*.json")):
            s = json.load(open(f))
            t["determinism"] += s["determinism"]
            t["shards"].append({k: s[k] for k in ("pool", "seed0", "n", "t_build_s", "t_total_s")})
        for pool, (s0, n) in POOLS.items():
            ents = []
            for i in range(n):
                r = json.load(open(f"{SH}/{task}_{pool}_{s0 + i}.json"))
                npy = f"{SH}/{task}_{pool}_{s0 + i}.npy"
                ents.append({"index": i, "pool": pool, "seed": s0 + i, "npy": npy, "npy_sha256": r["npy_sha256"],
                             "state_sha256": r["state_sha256"], "state_size": r["state_size"],
                             "objects": r["objects"], "robot_base_pos": r["robot_base_pos"],
                             "robot_base_quat": r["robot_base_quat"], "health": r["health"], "reset_s": r["reset_s"]})
            assert len({e["state_sha256"] for e in ents}) == n, "duplicate reset states"
            t["pools"][pool] = {"n": n, "seeds": [s0, s0 + n - 1], "entries": ents}
        bank["tasks"][task] = t
    body = json.dumps(bank, indent=1, sort_keys=True)
    bank["content_sha256_excluding_this_field"] = hashlib.sha256(body.encode()).hexdigest()
    json.dump(bank, open(OUT, "w"), indent=1)
    print("wrote", OUT, hashlib.sha256(open(OUT, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
