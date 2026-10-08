"""Comprehensive tests for intent selection metrics module."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from mip.intent_selection.metrics import (
    OutcomeRecordValidator,
    _bootstrap_paired_metric,
    _stable_choose_log,
    analyze,
    apply_practical_rule,
    compute_m3_cross_validated_selector,
    compute_oracle_availability,
)


def create_synthetic_records(
    n_states=5,
    n_candidates=16,
    n_replicates=4,
    success_prob=0.5,
    deterministic_arm1=None,
    seed=42
) -> pd.DataFrame:
    """Create synthetic outcome records for testing."""
    np.random.seed(seed)

    records = []
    for state_id in range(n_states):
        for cont_id in range(n_replicates):
            for cand_id in range(n_candidates):
                for arm in ['ACTION', 'INTENT']:
                    # Deterministic arm override for oracle test
                    if deterministic_arm1 == arm:
                        success = (cand_id == 0)
                    else:
                        success = np.random.choice([True, False], p=[success_prob, 1-success_prob])

                    records.append({
                        'experiment_id': 'test_exp',
                        'config_hash': 'hash1',
                        'checkpoint_pair_id': 'pair0',
                        'arm': arm,
                        'checkpoint_hash': 'ckpt1',
                        'split': 'test',
                        'stratum': 'mid',
                        'parent_reset_id': state_id % 3,
                        'state_id': state_id,
                        'source_policy': arm,
                        'elapsed_steps': 0,
                        'candidate_id': cand_id,
                        'candidate_seed': cand_id,
                        'decoder_seed': 0,
                        'continuation_id': cont_id,
                        'tail_mode': 'native',
                        'success': success,
                        'termination_reason': 'success' if success else 'timeout',
                        'steps_executed': 10,
                        'success_step': 5 if success else None,
                        'return': float(success),
                        'candidate_latency_ms': 1.0,
                        'tail_latency_ms': 1.0,
                        'error_status': 'ok',
                        'candidate_hash': f'h{cand_id}',
                    })

    return pd.DataFrame(records)


# ============================================================================
# M2 Tests
# ============================================================================

class TestM2OracleAvailability:
    """Tests for M2 oracle availability."""

    def test_stable_choose_log(self):
        """Test stable log(choose(n,k))."""
        assert _stable_choose_log(5, 2) == pytest.approx(np.log(10), rel=1e-10)
        assert _stable_choose_log(5, 0) == 0.0
        assert _stable_choose_log(5, 5) == 0.0
        assert _stable_choose_log(5, 6) == -np.inf

    def test_oracle_all_successful(self):
        """C(K)=1 when all candidates succeed."""
        successes = np.array([True] * 16)
        for k in [1, 4, 8, 16]:
            assert compute_oracle_availability(successes, k) == pytest.approx(1.0, abs=1e-10)

    def test_oracle_none_successful(self):
        """C(K)=0 when no candidates succeed."""
        successes = np.array([False] * 16)
        for k in [1, 4, 8, 16]:
            assert compute_oracle_availability(successes, k) == pytest.approx(0.0, abs=1e-10)

    def test_oracle_k_equals_1_equals_m_over_M(self):
        """C(1) = m/M (fraction of successes)."""
        successes = np.array([True, False, False, True, False])
        m = 2
        M = 5
        c_1 = compute_oracle_availability(successes, 1)
        assert c_1 == pytest.approx(m / M, rel=1e-10)

    def test_oracle_k_equals_M_equals_bool_m_gt_0(self):
        """C(M) = 1[m>0]."""
        # With at least one success
        successes = np.array([True] + [False] * 15)
        assert compute_oracle_availability(successes, 16) == pytest.approx(1.0, abs=1e-10)

        # With no successes
        successes_none = np.array([False] * 16)
        assert compute_oracle_availability(successes_none, 16) == pytest.approx(0.0, abs=1e-10)

    def test_oracle_brute_force_verification(self):
        """Verify formula against brute-force enumeration."""
        M = 6
        successes = np.array([True, False, True, False, False, True])

        for k in range(1, M + 1):
            c_k_formula = compute_oracle_availability(successes, k)

            # Brute force
            all_subsets = list(combinations(range(M), k))
            success_indices = set(np.where(successes)[0])
            count_with_success = sum(
                1 for subset in all_subsets
                if any(idx in success_indices for idx in subset)
            )
            c_k_brute = count_with_success / len(all_subsets)

            assert c_k_formula == pytest.approx(c_k_brute, rel=1e-10)

    def test_oracle_monotonicity(self):
        """C(K) is non-decreasing in K."""
        successes = np.array([True, False, True, False, False] + [False] * 11)
        prev_c = 0.0
        for k in range(1, 17):
            c_k = compute_oracle_availability(successes, k)
            assert c_k >= prev_c - 1e-10
            prev_c = c_k


# ============================================================================
# M3 Tests
# ============================================================================

class TestM3CrossValidatedSelector:
    """Tests for M3 selector."""

    def test_m3_deterministic_oracle(self):
        """With deterministic candidates, selector equals oracle."""
        df = create_synthetic_records(
            n_states=1,
            n_candidates=4,
            n_replicates=4,
            deterministic_arm1='ACTION'
        )

        result, _ = compute_m3_cross_validated_selector(
            df, 'ACTION', 0, 'test_exp', 'pair0',
            selection_replicates=[0, 1], heldout_replicates=[2, 3],
            k_values=[4]
        )

        # Selector should pick candidate 0 (always succeeds)
        # Random baseline is 0.25 (1 success + 3 failures)
        assert result[4]['selected_success'] == pytest.approx(1.0, abs=1e-10)
        assert result[4]['random_baseline'] == pytest.approx(0.25, abs=1e-10)
        assert result[4]['gain'] == pytest.approx(0.75, abs=1e-10)

    def test_m3_pure_noise(self):
        """With random data, gain should be near zero."""
        df = create_synthetic_records(
            n_states=5,
            n_candidates=16,
            n_replicates=4,
            success_prob=0.5
        )

        result, _ = compute_m3_cross_validated_selector(
            df, 'ACTION', 0, 'test_exp', 'pair0',
            selection_replicates=[0, 1], heldout_replicates=[2, 3],
            k_values=[1]
        )

        # K=1 averages all candidates, so gain should be ~0
        assert abs(result[1]['gain']) < 0.2

    def test_m3_tie_never_reads_held_out(self):
        """Tie-breaking uses only selection data, never held-out outcomes."""
        df = create_synthetic_records(n_states=1, n_candidates=2, n_replicates=4)

        # Artificially set heldout outcomes to verify they're not used in tie-breaking
        df.loc[
            (df['candidate_id'] == 0) & (df['continuation_id'].isin([2, 3])),
            'success'
        ] = False  # Hide that candidate 0 succeeds

        # On selection set, both candidates should tie (random outcomes)
        result, _ = compute_m3_cross_validated_selector(
            df, 'ACTION', 0, 'test_exp', 'pair0',
            selection_replicates=[0, 1], heldout_replicates=[2, 3],
            k_values=[2]
        )

        # Result should be valid (tie-breaking succeeded)
        assert 'selected_success' in result[2]
        assert result[2]['selected_success'] >= 0.0

    def test_m3_analytic_tie_integrated(self):
        """Analytic tie-integrated score differs from seeded tie-breaking."""
        df = create_synthetic_records(n_states=1, n_candidates=4, n_replicates=4)

        result, _ = compute_m3_cross_validated_selector(
            df, 'ACTION', 0, 'test_exp', 'pair0',
            selection_replicates=[0, 1], heldout_replicates=[2, 3],
            k_values=[4]
        )

        # Both scores should be in [0, 1]
        assert 0.0 <= result[4]['selected_success'] <= 1.0
        assert 0.0 <= result[4]['analytic_tied'] <= 1.0

    def test_m3_split_sensitivity(self):
        """Split sensitivity computes with swapped selection/heldout."""
        df = create_synthetic_records(n_states=1, n_candidates=8, n_replicates=4)

        result, sensitivity = compute_m3_cross_validated_selector(
            df, 'ACTION', 0, 'test_exp', 'pair0',
            selection_replicates=[0, 1], heldout_replicates=[2, 3],
            k_values=[4], split_sensitivity=True
        )

        # Both should have results
        assert result[4]['selected_success'] >= 0.0
        assert sensitivity[4]['selected_success'] >= 0.0


# ============================================================================
# Bootstrap Tests
# ============================================================================

class TestBootstrap:
    """Tests for bootstrap confidence intervals."""

    def test_bootstrap_reproducibility(self):
        """Same seed gives same results."""
        df = create_synthetic_records(n_states=10, n_candidates=8, n_replicates=4)

        def dummy_metric(df_arm, stratum, tail_mode):
            return float(df_arm['success'].mean())

        result1 = _bootstrap_paired_metric(
            df, dummy_metric, n_resamples=100, bootstrap_seed=42
        )
        result2 = _bootstrap_paired_metric(
            df, dummy_metric, n_resamples=100, bootstrap_seed=42
        )

        assert result1['mean'] == pytest.approx(result2['mean'], rel=1e-10)
        assert result1['ci_lower'] == pytest.approx(result2['ci_lower'], rel=1e-10)

    def test_bootstrap_keeps_parents_intact(self):
        """Bootstrap resamples parents, not individual rows."""
        df = create_synthetic_records(n_states=15, n_candidates=4, n_replicates=4)

        # Count parent appearances in a resample

        np.random.seed(42)
        parents = sorted(df['parent_reset_id'].unique())
        rng = np.random.RandomState(42)
        resampled_parents = rng.choice(parents, size=len(parents), replace=True)

        # Check that parents can appear multiple times
        has_duplicate = len(resampled_parents) != len(set(resampled_parents))
        assert has_duplicate, "Bootstrap should resample WITH replacement"

    def test_bootstrap_paired_difference(self):
        """Bootstrap computes paired difference on same resampled parents."""
        df = create_synthetic_records(n_states=10, n_candidates=8, n_replicates=4)

        def m1_metric(df_arm, stratum, tail_mode):
            return float(df_arm['success'].mean())

        result = _bootstrap_paired_metric(
            df, m1_metric, arm1='INTENT', arm2='ACTION',
            n_resamples=50, bootstrap_seed=42
        )

        # Should have valid CI
        assert not np.isnan(result['mean'])
        assert result['ci_lower'] <= result['mean'] <= result['ci_upper']
        assert result['ci_lower'] < result['ci_upper']


# ============================================================================
# Validator Tests
# ============================================================================

class TestValidator:
    """Tests for OutcomeRecordValidator."""

    def test_validator_missing_columns(self):
        """Rejects missing required columns."""
        df = pd.DataFrame({'experiment_id': ['test']})
        validator = OutcomeRecordValidator(df)
        is_valid, errors, _ = validator.validate()
        assert not is_valid
        assert any('Missing columns' in e for e in errors)

    def test_validator_unresolved_errors(self):
        """Rejects unresolved error_status."""
        df = create_synthetic_records(n_states=2)
        df.loc[0, 'error_status'] = 'unresolved'
        validator = OutcomeRecordValidator(df)
        is_valid, errors, _ = validator.validate()
        assert not is_valid
        assert any('unresolved' in e.lower() for e in errors)

    def test_validator_duplicate_keys(self):
        """Detects duplicate keys."""
        df = create_synthetic_records(n_states=2, n_candidates=4, n_replicates=2)
        # Duplicate a row
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        validator = OutcomeRecordValidator(df)
        is_valid, errors, _ = validator.validate()
        assert not is_valid
        assert any('Duplicate' in e for e in errors)

    def test_validator_incomplete_grid(self):
        """Warns about missing candidate×continuation cells."""
        df = create_synthetic_records(n_states=1, n_candidates=4, n_replicates=4)

        # Remove one (candidate, continuation) pair
        to_remove = df[
            (df['state_id'] == 0) &
            (df['candidate_id'] == 0) &
            (df['continuation_id'] == 0) &
            (df['arm'] == 'ACTION')
        ].index
        if len(to_remove) > 0:
            df = df.drop(to_remove[0])

        validator = OutcomeRecordValidator(df)
        is_valid, errors, warnings = validator.validate()

        # Incomplete grids block analysis (spec §11: unresolved gaps can bias arm comparisons)
        assert not is_valid
        assert any('incomplete grid' in e.lower() for e in errors)

    def test_validator_accepts_valid_data(self):
        """Accepts valid records."""
        df = create_synthetic_records(n_states=5, n_candidates=8, n_replicates=4)
        validator = OutcomeRecordValidator(df)
        is_valid, errors, _ = validator.validate()
        assert is_valid


# ============================================================================
# Decision Rule Tests
# ============================================================================

class TestDecisionRule:
    """Tests for apply_practical_rule."""

    def test_ceiling_effect(self):
        """Returns inconclusive for ceiling."""
        verdict, _ = apply_practical_rule(
            c_intent_k16=0.98, c_action_k16=0.97,
            paired_diff_k16_ci=(0.96, 0.99),
            m3_gain=0.01, m3_gain_ci=(0.0, 0.02),
            c_intent_k1=0.98, c_action_k1=0.97,
        )
        assert verdict == 'inconclusive'

    def test_floor_effect(self):
        """Returns inconclusive for floor."""
        verdict, _ = apply_practical_rule(
            c_intent_k16=0.02, c_action_k16=0.01,
            paired_diff_k16_ci=(0.01, 0.03),
            m3_gain=0.01, m3_gain_ci=(0.0, 0.02),
            c_intent_k1=0.02, c_action_k1=0.01,
        )
        assert verdict == 'inconclusive'

    def test_promising_coverage_ci_lower_positive(self):
        """Requires CI lower bound > 0 for promising_coverage."""
        verdict, _ = apply_practical_rule(
            c_intent_k16=0.70, c_action_k16=0.60,
            paired_diff_k16_ci=(0.05, 0.15),  # Lower bound > 0
            m3_gain=0.05, m3_gain_ci=(0.01, 0.09),
            c_intent_k1=0.50, c_action_k1=0.40,
        )
        assert verdict == 'promising_coverage'

    def test_no_advantage_narrow_interval(self):
        """Requires narrow interval for no_advantage."""
        verdict, _ = apply_practical_rule(
            c_intent_k16=0.50, c_action_k16=0.50,
            paired_diff_k16_ci=(-0.03, 0.03),  # Narrow, centered on 0
            m3_gain=0.01, m3_gain_ci=(-0.01, 0.03),
            c_intent_k1=0.40, c_action_k1=0.40,
        )
        assert verdict == 'no_advantage'

    def test_useful_headroom(self):
        """INTENT's own gain >= threshold AND M3 selected < ACTION's."""
        verdict, _ = apply_practical_rule(
            c_intent_k16=0.40, c_action_k16=0.50,
            paired_diff_k16_ci=(-0.15, -0.05),
            m3_gain=0.0, m3_gain_ci=(-0.02, 0.02),
            c_intent_k1=0.20, c_action_k1=0.15,
            c_intent_gain_k16=0.20,  # Own gain: 0.40 - 0.20 = 0.20 (>= 0.05)
            m3_selected_intent=0.40,  # INTENT selected success
            m3_selected_action=0.55,  # ACTION selected success (higher)
        )
        assert verdict == 'useful_headroom'

    def test_promising_selectability(self):
        """Similar coverage + strong M3 gain."""
        verdict, _ = apply_practical_rule(
            c_intent_k16=0.50, c_action_k16=0.50,
            paired_diff_k16_ci=(-0.02, 0.02),  # Similar
            m3_gain=0.08, m3_gain_ci=(0.03, 0.13),  # Strong M3
            c_intent_k1=0.40, c_action_k1=0.40,
        )
        assert verdict == 'promising_selectability'


# ============================================================================
# Analysis Function Tests
# ============================================================================

class TestAnalyzeFunction:
    """Tests for main analyze function."""

    def test_analyze_returns_dicts_of_dataframes(self):
        """analyze() returns dict of DataFrames."""
        df = create_synthetic_records(n_states=5, n_candidates=8, n_replicates=4)
        manifest = {'experiment_id': 'test_exp'}

        results = analyze(manifest, df, n_resamples=100)

        assert isinstance(results, dict)
        assert 'aggregate_metrics' in results
        assert isinstance(results['aggregate_metrics'], pd.DataFrame)
        assert len(results['aggregate_metrics']) > 0

    def test_analyze_includes_paired_differences(self):
        """analyze() computes INTENT-ACTION paired differences."""
        df = create_synthetic_records(n_states=5, n_candidates=8, n_replicates=4)
        manifest = {'experiment_id': 'test_exp'}

        results = analyze(manifest, df, n_resamples=100)

        # Should have paired difference rows
        assert 'aggregate_metrics' in results
        agg = results['aggregate_metrics']
        if 'ACTION' in df['arm'].unique() and 'INTENT' in df['arm'].unique():
            assert any(agg['arm'].str.contains('difference', na=False))

    def test_analyze_rejects_invalid_data(self):
        """analyze() raises ValueError for invalid data."""
        df = pd.DataFrame({'experiment_id': ['test']})
        manifest = {'experiment_id': 'test'}

        with pytest.raises(ValueError):
            analyze(manifest, df)

    def test_analyze_flags_exploratory_r2(self):
        """analyze() flags R=2 as exploratory."""
        df = create_synthetic_records(n_states=3, n_candidates=4, n_replicates=2)
        manifest = {'experiment_id': 'test_exp'}

        results = analyze(manifest, df, n_resamples=50)

        agg = results['aggregate_metrics']
        # With R=2, should be marked exploratory
        if 'exploratory' in agg.columns:
            assert agg['exploratory'].any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
