"""Unit tests for the oops spatial-robustness pipeline (no LIBERO / GPU needed).

  pytest examples/openpi/oops/oops_test.py
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import oops_common as oc  # noqa: E402
from build_case_manifest import classify  # noqa: E402


def test_case_grid_counts_delta0_once():
    cases = oc.physical_cases([0, 1, 2], [0.0, 0.05, 0.10], 10)
    assert len(cases) == 3 * (1 + 10 + 10)          # 63 per task, 504 for 8 tasks
    assert sum(1 for c in cases if c[1] == 0.0) == 3


def test_perturb_moves_only_xy_of_target():
    s = np.arange(79, dtype=np.float64)
    p = oc.perturb_state(s, 10, 0.03, -0.04)
    diff = np.nonzero(p - s)[0].tolist()
    assert diff == [10, 11]
    assert math.isclose(p[10] - s[10], 0.03) and math.isclose(p[11] - s[11], -0.04)
    assert s[10] == 10.0  # input untouched


def test_displacement_magnitude_and_fixed_angles():
    for a in oc.angles(10):
        dx, dy = oc.displacement(0.05, a)
        assert math.isclose(math.hypot(dx, dy), 0.05)
    assert oc.displacement(0.0, float("nan")) == (0.0, 0.0)
    assert np.allclose(oc.angles(10), np.arange(10) * 2 * math.pi / 10)


def test_episode_seed_deterministic_and_order_independent():
    a = oc.episode_seed("put the bowl on the stove", 1, 0.05, 3, 0)
    assert a == oc.episode_seed("Put the bowl on the stove ", 1, 0.05, 3, 0)
    assert a != oc.episode_seed("put the bowl on the stove", 1, 0.05, 4, 0)
    assert 0 <= a < 2 ** 31 - 1


def test_case_id_contains_identity():
    h = oc.state_hash(np.zeros(3))
    cid = oc.case_id("put the bowl on the plate", 2, 0.1, 7, h)
    assert cid.startswith("put_the_bowl_on_the_plate|r2|d0.100|a7|") and h[:12] in cid
    assert oc.CONFIG_VERSION in cid


def test_jsonl_tolerates_torn_last_line(tmp_path):
    p = tmp_path / "x.jsonl"
    oc.append_jsonl(p, {"a": 1})
    with open(p, "a") as f:
        f.write('{"a": 2')  # killed mid-write
    assert oc.read_jsonl(p) == [{"a": 1}]


def _m(**kw):
    base = {"pre_satisfied": False, "on_table_footprint": True, "placement_penetration_m": 0.0, "xy_drift_m": 0.0,
            "z_change_m": 0.0, "other_disturb_max_m": 0.0, "post_settle_motion_m": 0.0}
    base.update(kw)
    return base


def test_classify_rules():
    R = oc.VALIDITY_RULES
    assert classify(_m(), R, False) == ("valid", ["ok"])
    assert classify(_m(pre_satisfied=True), R, False)[0] == "invalid"
    assert classify(_m(xy_drift_m=0.05), R, False) == ("invalid", ["displaced_by_contact"])
    assert classify(_m(xy_drift_m=0.03), R, False) == ("unknown", ["xy_drift_borderline"])
    assert classify(_m(z_change_m=-0.05), R, False)[1] == ["fell"]
    assert classify(_m(z_change_m=0.05), R, False)[1] == ["support_changed_elevated"]
    assert classify(_m(placement_penetration_m=0.01), R, False)[1] == ["interpenetration"]
    assert classify(_m(other_disturb_max_m=0.03), R, False)[1] == ["scene_disturbed"]
    assert classify({"sim_exception": "x"}, R, False) == ("unknown", ["sim_exception"])
    # delta=0: drift-type checks are relative to itself and skipped
    assert classify(_m(xy_drift_m=0.05), R, True) == ("valid", ["ok"])


def test_square_dist_outside_demo_box():
    from square_build_manifest import dist_outside_box
    box = {"x": [-0.115, -0.110], "y": [0.111, 0.225]}
    assert dist_outside_box([-0.112, 0.15], box) == 0.0
    assert math.isclose(dist_outside_box([-0.065, 0.15], box), 0.045)
    assert math.isclose(dist_outside_box([-0.110 + 0.03, 0.225 + 0.04], box), 0.05)


def test_square_adapter_seeds_native_reset_before_reset_to():
    from square_eval import OneEnvAdapter
    calls = []

    class Robo:
        def reset(self):
            calls.append(("reset", np.random.randint(1 << 30)))

    class Low:
        init_state = None

    class Env:
        def __init__(self, low):
            self.low = low

        def reset(self):
            calls.append(("reset_to", self.low.init_state is not None))
            return np.zeros((2, 23)), {}

    low = Low()
    ad = OneEnvAdapter(Env(low), low, Robo())
    ad.begin(np.zeros(45), 42)
    obs, _ = ad.reset()
    np.random.seed(42)
    assert calls == [("reset", np.random.randint(1 << 30)), ("reset_to", True)]
    assert obs.shape == (1, 2, 23) and low.init_state is None
