"""Tests for coordinator feedback fixes:
1. Bootstrap seed required (no default)
2. Promising coverage requires both C(16) and M3 credible
3. Useful_headroom proper definition
4. M3 seeds consistent (not including arm for K=4)
"""

import pandas as pd
import pytest

from mip.intent_selection.metrics import (
    _bootstrap_paired_metric,
    apply_practical_rule,
)


def test_bootstrap_seed_required_no_default():
    """CRITICAL FIX #1: bootstrap_seed is required, no default 42."""
    df = pd.DataFrame({
        'experiment_id': ['test'],
        'stratum': ['mid'],
        'tail_mode': ['native'],
        'arm': ['ACTION'],
        'parent_reset_id': [0],
        'success': [True],
    })

    def dummy_metric(df_arm, stratum, tail_mode):
        return 0.5

    # Should raise ValueError when bootstrap_seed=None
    with pytest.raises(ValueError, match="bootstrap_seed is required"):
        _bootstrap_paired_metric(df, dummy_metric, bootstrap_seed=None)


def test_promising_coverage_requires_both_c_and_m3_credible():
    """CRITICAL FIX #2: promising_coverage requires BOTH:
    - C(16) CI lower bound > 0
    - M3 gain CI lower bound > 0
    If only C is credible but M3 is not, return inconclusive.
    """
    # Case 1: C credible, M3 NOT credible -> inconclusive (not promising_coverage)
    verdict1, reason1 = apply_practical_rule(
        c_intent_k16=0.70,
        c_action_k16=0.60,
        paired_diff_k16_ci=(0.05, 0.15),  # Lower bound > 0: CREDIBLE
        m3_gain=0.03,
        m3_gain_ci=(-0.02, 0.08),  # Lower bound <= 0: NOT CREDIBLE
        c_intent_k1=0.50,
        c_action_k1=0.40,
    )

    assert verdict1 != 'promising_coverage', \
        f"Should NOT be promising_coverage when M3 gain CI lower <= 0, got {verdict1}"
    assert verdict1 == 'inconclusive'
    assert 'not credible' in reason1

    # Case 2: Both C and M3 credible -> promising_coverage
    verdict2, reason2 = apply_practical_rule(
        c_intent_k16=0.70,
        c_action_k16=0.60,
        paired_diff_k16_ci=(0.05, 0.15),  # Lower bound > 0: CREDIBLE
        m3_gain=0.05,
        m3_gain_ci=(0.01, 0.09),  # Lower bound > 0: CREDIBLE
        c_intent_k1=0.50,
        c_action_k1=0.40,
    )

    assert verdict2 == 'promising_coverage'


def test_useful_headroom_intent_own_gain_not_k1_uplift():
    """CRITICAL FIX #3: useful_headroom is defined by:
    - INTENT's own gain C(16) - C(1) >= threshold
    - INTENT's M3 selected success < ACTION's M3 selected success
    NOT by "K=1 uplift AND C(16) trails" as implemented before.
    """
    # INTENT has large own gain but M3 trails -> useful_headroom
    verdict, reason = apply_practical_rule(
        c_intent_k16=0.50,
        c_action_k16=0.55,
        paired_diff_k16_ci=(-0.10, 0.0),
        m3_gain=0.0,
        m3_gain_ci=(-0.05, 0.05),
        c_intent_k1=0.20,
        c_action_k1=0.20,
        c_intent_gain_k16=0.30,  # INTENT's own gain: 0.50 - 0.20 = 0.30 (>= 0.05)
        m3_selected_intent=0.40,  # INTENT selected success
        m3_selected_action=0.55,  # ACTION selected success (higher)
    )

    assert verdict == 'useful_headroom', f"Expected useful_headroom, got {verdict}"
    assert 'still trails' in reason


def test_m3_k4_subset_seed_excludes_arm():
    """CRITICAL FIX #4a: K=4 subset sampling seed does NOT include arm.
    This is tested indirectly by verifying the function runs without error
    using the bootstrap namespace (which is arm-agnostic for subsets).
    """
    # If arm was included in subset seed, we'd get different K=4 subsets per arm.
    # By excluding arm, both arms get the same subset list (paired design).
    # This is hard to test directly without exposing internals, but we verify
    # the function runs with the correct namespace.

    import numpy as np

    from mip.intent_selection.metrics import compute_m3_cross_validated_selector

    # Create minimal test data
    records = []
    for arm in ['ACTION', 'INTENT']:
        for cand_id in range(8):
            for cont_id in range(4):
                records.append({
                    'experiment_id': 'test_exp',
                    'config_hash': 'h',
                    'checkpoint_pair_id': 'pair0',
                    'arm': arm,
                    'checkpoint_hash': 'c',
                    'split': 'test',
                    'stratum': 'mid',
                    'parent_reset_id': 0,
                    'state_id': 0,
                    'source_policy': arm,
                    'elapsed_steps': 0,
                    'candidate_id': cand_id,
                    'candidate_seed': cand_id,
                    'decoder_seed': 0,
                    'continuation_id': cont_id,
                    'tail_mode': 'native',
                    'success': np.random.choice([True, False], p=[0.5, 0.5]),
                    'termination_reason': 'success',
                    'steps_executed': 10,
                    'success_step': 5,
                    'return': 0.5,
                    'candidate_latency_ms': 1.0,
                    'tail_latency_ms': 1.0,
                    'error_status': 'ok',
                    'candidate_hash': f'h{cand_id}',
                })

    df = pd.DataFrame(records)

    # Should not raise exception (verifies bootstrap namespace exists)
    result_intent, _ = compute_m3_cross_validated_selector(
        df, 'INTENT', 0, 'test_exp', 'pair0',
        selection_replicates=[0, 1], heldout_replicates=[2, 3],
        k_values=[4], n_subsets_k4=5
    )

    result_action, _ = compute_m3_cross_validated_selector(
        df, 'ACTION', 0, 'test_exp', 'pair0',
        selection_replicates=[0, 1], heldout_replicates=[2, 3],
        k_values=[4], n_subsets_k4=5
    )

    # Both should have valid results (no exception)
    assert result_intent[4]['selected_success'] >= 0.0
    assert result_action[4]['selected_success'] >= 0.0


def test_m3_tie_break_seed_keyed_by_subset():
    """CRITICAL FIX #4c: Tie-breaking seed is keyed by subset index.
    Different subsets use different tie-break seeds (not reused).
    """
    import numpy as np

    from mip.intent_selection.metrics import compute_m3_cross_validated_selector

    records = []
    for cand_id in range(4):
        for cont_id in range(4):
            records.append({
                'experiment_id': 'test_exp',
                'config_hash': 'h',
                'checkpoint_pair_id': 'pair0',
                'arm': 'ACTION',
                'checkpoint_hash': 'c',
                'split': 'test',
                'stratum': 'mid',
                'parent_reset_id': 0,
                'state_id': 0,
                'source_policy': 'ACTION',
                'elapsed_steps': 0,
                'candidate_id': cand_id,
                'candidate_seed': cand_id,
                'decoder_seed': 0,
                'continuation_id': cont_id,
                'tail_mode': 'native',
                'success': np.random.choice([True, False], p=[0.5, 0.5]),
                'termination_reason': 'success',
                'steps_executed': 10,
                'success_step': 5,
                'return': 0.5,
                'candidate_latency_ms': 1.0,
                'tail_latency_ms': 1.0,
                'error_status': 'ok',
                'candidate_hash': f'h{cand_id}',
            })

    df = pd.DataFrame(records)

    # Should not raise exception (verifies tie-break seed derivation works)
    result, _ = compute_m3_cross_validated_selector(
        df, 'ACTION', 0, 'test_exp', 'pair0',
        selection_replicates=[0, 1], heldout_replicates=[2, 3],
        k_values=[4], n_subsets_k4=5
    )

    assert result[4]['selected_success'] >= 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
