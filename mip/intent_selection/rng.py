"""Stable, keyed RNG derivation for reproducible experiments.

Uses cryptographic hashing (blake2b/sha256) instead of Python's built-in hash()
to ensure reproducibility across Python versions and implementations.

Implements separate namespaces for different RNG uses:
- state_collection: capturing environment states
- candidate: sampling candidate policies
- decoder: decoder noise (fixed across candidates at each state)
- continuation: continuation policy sampling
- env: environment stochasticity
- bootstrap: bootstrap resampling
- video: video selection

All derivations are deterministic and independent of process state.
"""

from __future__ import annotations

import hashlib


class KeyedRNG:
    """Deterministic RNG seed derivation using stable hashing.

    Never uses Python's built-in hash() which is randomized.
    Uses blake2b/sha256 for stable, reproducible key derivation.
    """

    # Named namespaces for different uses
    NAMESPACES = {
        "state_collection": 0,
        "candidate": 1,
        "decoder": 2,
        "continuation": 3,
        "env": 4,
        "bootstrap": 5,
        "video": 6,
        "m3_subsets": 7,
        "m3_ties": 8,
    }

    REQUIRED_KEYS = {
        "state_collection": ("state_id",),
        "candidate": ("state_id", "candidate_id"),
        "decoder": ("state_id", "candidate_id"),
        "continuation": ("state_id", "continuation_id"),
        "env": ("state_id",),
        "bootstrap": ("state_id",),
        "video": ("state_id", "candidate_id"),
        "m3_subsets": ("state_id",),
        "m3_ties": ("state_id", "subset_index"),
    }

    # Continuation randomness is deliberately shared across candidates (spec §7).
    FORBIDDEN_KEYS = {"continuation": ("candidate_id",)}

    def __init__(self, experiment_id: str):
        """Initialize keyed RNG with an experiment ID.

        Args:
            experiment_id: Base experiment identifier (e.g., "intent_selection_square_mh_v1")
        """
        self.experiment_id = experiment_id

    def derive_seed(
        self,
        namespace: str,
        *,
        experiment: str | None = None,
        arm: str | None = None,
        checkpoint_pair: str | None = None,
        state_id: int | None = None,
        candidate_id: int | None = None,
        continuation_id: int | None = None,
        replan_index: int | None = None,
        policy_identity: str | None = None,
        subset_index: int | None = None,
    ) -> int:
        """Derive a deterministic seed from hierarchical keys.

        Seed depends on the specified keys according to the namespace:
        - state_collection, candidate, decoder: state_id + candidate_id
        - continuation: state_id + continuation_id + replan_index (NO candidate_id)
        - env, bootstrap: state_id only
        - video: state_id + candidate_id

        Args:
            namespace: One of NAMESPACES keys
            experiment: Experiment name (if different from self.experiment_id)
            arm: Policy arm name (e.g., "ACTION", "INTENT")
            checkpoint_pair: Checkpoint pair identifier (e.g., "seed_pair_0")
            state_id: Episode/state identifier
            candidate_id: Candidate index within a state
            continuation_id: Continuation replicate ID
            replan_index: Index within the continuation horizon
            policy_identity: Policy name for continuation

        Returns:
            Deterministic integer seed (0-2^31-1)

        Raises:
            ValueError: If namespace is unknown or required keys are missing
        """
        if namespace not in self.NAMESPACES:
            raise ValueError(
                f"Unknown namespace: {namespace}. "
                f"Must be one of {list(self.NAMESPACES.keys())}"
            )

        exp = experiment or self.experiment_id

        keys = {
            "arm": arm,
            "checkpoint_pair": checkpoint_pair,
            "state_id": state_id,
            "candidate_id": candidate_id,
            "continuation_id": continuation_id,
            "replan_index": replan_index,
            "policy_identity": policy_identity,
            "subset_index": subset_index,
        }
        for required in self.REQUIRED_KEYS[namespace]:
            if keys[required] is None:
                raise ValueError(f"{namespace} requires {required}")
        for forbidden in self.FORBIDDEN_KEYS.get(namespace, ()):
            if keys[forbidden] is not None:
                raise ValueError(f"{namespace} must not depend on {forbidden}")

        # Every supplied key enters the hash; nothing is silently dropped.
        key_parts = [exp, namespace] + [
            f"{name}={value}" for name, value in keys.items() if value is not None
        ]

        # Hash the key components
        key_string = "|".join(str(p) for p in key_parts)
        hash_bytes = hashlib.blake2b(
            key_string.encode("utf-8"),
            digest_size=8,  # 64 bits
        ).digest()

        # Convert to seed (0-2^31-1)
        seed = int.from_bytes(hash_bytes[:4], byteorder="little") & 0x7FFFFFFF

        return seed

    @staticmethod
    def derive_seed_global(
        namespace: str,
        experiment_id: str,
        **kwargs
    ) -> int:
        """Static method for single-call seed derivation.

        Args:
            namespace: One of KeyedRNG.NAMESPACES
            experiment_id: Experiment identifier
            **kwargs: Arguments to pass to derive_seed()

        Returns:
            Deterministic integer seed
        """
        rng = KeyedRNG(experiment_id)
        return rng.derive_seed(namespace, **kwargs)


def test_determinism():
    """Test that seed derivation is deterministic."""
    rng1 = KeyedRNG("test_exp")
    rng2 = KeyedRNG("test_exp")

    # Same inputs should produce same seed
    seed1 = rng1.derive_seed(
        "candidate",
        arm="ACTION",
        checkpoint_pair="seed_pair_0",
        state_id=42,
        candidate_id=5,
    )
    seed2 = rng2.derive_seed(
        "candidate",
        arm="ACTION",
        checkpoint_pair="seed_pair_0",
        state_id=42,
        candidate_id=5,
    )
    assert seed1 == seed2, f"Determinism failed: {seed1} != {seed2}"

    # Different state_id should produce different seeds
    seed3 = rng1.derive_seed(
        "candidate",
        arm="ACTION",
        checkpoint_pair="seed_pair_0",
        state_id=43,
        candidate_id=5,
    )
    assert seed1 != seed3, "Different state_id should produce different seeds"

    # Continuation seeds should NOT depend on candidate_id
    cont_seed1 = rng1.derive_seed(
        "continuation",
        state_id=42,
        continuation_id=0,
        policy_identity="ACTION",
    )
    cont_seed2 = rng1.derive_seed(
        "continuation",
        state_id=42,
        continuation_id=1,
        policy_identity="ACTION",
    )
    assert cont_seed1 != cont_seed2, "Different continuation_id should produce different seeds"

    print("✓ All determinism tests passed")


if __name__ == "__main__":
    test_determinism()
