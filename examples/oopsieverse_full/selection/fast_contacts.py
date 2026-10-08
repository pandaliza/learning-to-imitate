"""Performance shim for DamageSim's RSMechanicalDamageEvaluator._get_part_contacts (selection harness only).

The released method (damagesim/robosuite/evaluators/mechanical.py:75-111) rebuilds, for every tracked body on every
control step, the set of the body's geoms with a Python loop over all model geoms (1,709 in the pick_egg kitchen,
x16 tracked bodies, each element read through the robosuite binding) and then scans every contact. That is ~1/3
of the env step time. This shim computes the identical geom set with one numpy comparison and scans the contacts in
the same order with the same arithmetic, returning the same list of dicts (same keys, values, element types and
order). Nothing else changes; the official file is untouched (class attribute patched at import, like the official
patch_gripper_for_position_control). Equivalence is verified per call, on damaging trajectories, by
verify_fast_contacts.py (outputs/oopsieverse_full_study/selection/verification/fast_contacts.json).

Diff vs the released method (logic):
-        body_geoms = set()
-        for gid in range(model.ngeom):
-            if model.geom_bodyid[gid] == bid:
-                body_geoms.add(gid)
+        body_geoms = set(np.nonzero(np.asarray(model.geom_bodyid) == bid)[0].tolist())
(the contact loop below it is kept verbatim.)
"""

import numpy as np

_ORIG = None


def fast_get_part_contacts(self, part_name):
    sim = self._sim()
    bid = self._body_id(part_name)
    if sim is None or bid is None:
        return []
    model = sim.model
    data = sim.data
    body_geoms = set(np.nonzero(np.asarray(model.geom_bodyid) == bid)[0].tolist())
    if not body_geoms:
        return []
    contacts = []
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.geom1 not in body_geoms and contact.geom2 not in body_geoms:
            continue
        efc = contact.efc_address
        if efc < 0:
            continue
        normal_force = abs(data.efc_force[efc])
        normal_dir = contact.frame[:3]
        force_vec = normal_force * normal_dir
        if normal_force > 1e-6:
            contacts.append({
                "body_id": bid,
                "force": force_vec.copy(),
                "magnitude": float(normal_force),
                "geom1": contact.geom1,
                "geom2": contact.geom2,
            })
    return contacts


def install():
    global _ORIG
    from damagesim.robosuite.evaluators.mechanical import RSMechanicalDamageEvaluator
    if _ORIG is None:
        _ORIG = RSMechanicalDamageEvaluator._get_part_contacts
    RSMechanicalDamageEvaluator._get_part_contacts = fast_get_part_contacts
    return _ORIG


def uninstall():
    from damagesim.robosuite.evaluators.mechanical import RSMechanicalDamageEvaluator
    if _ORIG is not None:
        RSMechanicalDamageEvaluator._get_part_contacts = _ORIG


def original():
    from damagesim.robosuite.evaluators.mechanical import RSMechanicalDamageEvaluator
    return _ORIG if _ORIG is not None else RSMechanicalDamageEvaluator._get_part_contacts
