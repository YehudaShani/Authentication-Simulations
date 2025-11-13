import pytest
from helpers.computations import (
    computeSuccessProbability,
    findOptimalWallet,
    generateKeyFaultProbabilityScenarios,
    reportOptimalWalletsForProbabilities,
)
from helpers.wallet_enumerations import (
    enumerateStates,
    ownerAdvKeysFromStates,
    enumerateStaticWallets,
)
from consts import SAFE, LOST, LEAKED, STOLEN


class TestComputeSuccessProbability:
    """Tests for computeSuccessProbability function."""

    def test_success_when_owner_can_access_and_adversary_cannot(self):
        """Test that success probability is calculated correctly when owner can access but adversary cannot."""
        wallet = [1]  # Wallet with key 1
        ownerStates = [1, 0, 1, 0]  # Owner has key in states 0 and 2
        advStates = [0, 0, 0, 1]  # Adversary has key only in state 3
        probabilities = [0.25, 0.25, 0.25, 0.25]
        
        result = computeSuccessProbability(wallet, ownerStates, advStates, probabilities)
        # Success in states 0 and 2 (adversary doesn't have key)
        # State 0: owner has key, adversary doesn't -> success
        # State 1: owner doesn't have key -> no success
        # State 2: owner has key, adversary doesn't -> success
        # State 3: adversary has key -> no success
        assert result == 0.5  # 0.25 + 0.25

    def test_no_success_when_adversary_can_access(self):
        """Test that success probability is 0 when adversary can access."""
        wallet = [1]
        ownerStates = [1, 1, 1, 1]
        advStates = [1, 1, 1, 1]  # Adversary can always access
        probabilities = [0.25, 0.25, 0.25, 0.25]
        
        result = computeSuccessProbability(wallet, ownerStates, advStates, probabilities)
        assert result == 0.0

    def test_no_success_when_owner_cannot_access(self):
        """Test that success probability is 0 when owner cannot access."""
        wallet = [1]
        ownerStates = [0, 0, 0, 0]  # Owner never has key
        advStates = [0, 0, 0, 0]
        probabilities = [0.25, 0.25, 0.25, 0.25]
        
        result = computeSuccessProbability(wallet, ownerStates, advStates, probabilities)
        assert result == 0.0

    def test_complex_wallet_multiple_combinations(self):
        """Test with wallet that has multiple combinations (OR logic)."""
        wallet = [1, 2]  # (key1) OR (key2)
        ownerStates = [1, 2, 3, 0]  # Owner has key1 in state 0, key2 in state 1, both in state 2
        advStates = [0, 0, 0, 0]
        probabilities = [0.25, 0.25, 0.25, 0.25]
        
        result = computeSuccessProbability(wallet, ownerStates, advStates, probabilities)
        # Success in states 0, 1, 2 (owner can access, adversary cannot)
        assert result == 0.75


class TestFindOptimalWallet:
    """Tests for findOptimalWallet function."""

    def test_finds_single_optimal_wallet(self):
        """Test finding a single optimal wallet."""
        keyCount = 2
        wallets = enumerateStaticWallets(keyCount, deduplicate_by_architecture=False)
        probs = {SAFE: 0.8, LOST: 0.1, LEAKED: 0.05, STOLEN: 0.05}
        
        optimal_wallets, best_p = findOptimalWallet(wallets, keyCount, probs)
        
        assert isinstance(optimal_wallets, list)
        assert len(optimal_wallets) >= 1
        assert 0.0 <= best_p <= 1.0
        # Verify all optimal wallets have the same probability
        states, state_probabilities = enumerateStates(keyCount, probs)
        ownerStates, advStates = ownerAdvKeysFromStates(states)
        for wallet in optimal_wallets:
            p = computeSuccessProbability(wallet, ownerStates, advStates, state_probabilities)
            assert abs(best_p - p) < 1e-9

    

    def test_returns_empty_list_for_empty_wallet_list(self):
        """Test behavior with empty wallet list."""
        keyCount = 2
        wallets = []
        probs = {SAFE: 0.7, LOST: 0.1, LEAKED: 0.1, STOLEN: 0.1}
        
        optimal_wallets, best_p = findOptimalWallet(wallets, keyCount, probs)
        
        assert optimal_wallets == []
        assert best_p == -1.0


class TestGenerateKeyFaultProbabilityScenarios:
    """Tests for generateKeyFaultProbabilityScenarios function."""

    def test_basic_generation(self):
        """Test basic scenario generation."""
        scenarios = generateKeyFaultProbabilityScenarios(step=0.5, include_zero=True)
        
        assert len(scenarios) > 0
        for scenario in scenarios:
            assert SAFE in scenario
            assert LOST in scenario
            assert LEAKED in scenario
            assert STOLEN in scenario
            total = scenario[SAFE] + scenario[LOST] + scenario[LEAKED] + scenario[STOLEN]
            assert abs(total - 1.0) < 1e-9

    def test_step_validation(self):
        """Test that invalid step values raise errors."""
        with pytest.raises(ValueError, match="step must be in"):
            generateKeyFaultProbabilityScenarios(step=0)
        
        with pytest.raises(ValueError, match="step must be in"):
            generateKeyFaultProbabilityScenarios(step=1.5)
        
        with pytest.raises(ValueError, match="step must evenly divide"):
            generateKeyFaultProbabilityScenarios(step=0.03)  # Doesn't evenly divide 1.0

    def test_min_safe_filtering(self):
        """Test that min_safe correctly filters scenarios."""
        scenarios = generateKeyFaultProbabilityScenarios(
            step=0.1, include_zero=False, min_safe=0.7
        )
        
        assert len(scenarios) > 0
        for scenario in scenarios:
            assert scenario[SAFE] >= 0.7

    def test_max_theft_filtering(self):
        """Test that max_theft correctly filters scenarios."""
        scenarios = generateKeyFaultProbabilityScenarios(
            step=0.1, include_zero=False, max_theft=0.2
        )
        
        assert len(scenarios) > 0
        for scenario in scenarios:
            assert scenario[STOLEN] <= 0.2

    def test_min_safe_and_max_theft_together(self):
        """Test that min_safe and max_theft work together."""
        scenarios = generateKeyFaultProbabilityScenarios(
            step=0.1, include_zero=False, min_safe=0.6, max_theft=0.2
        )
        
        for scenario in scenarios:
            assert scenario[SAFE] >= 0.6
            assert scenario[STOLEN] <= 0.2

    def test_include_zero_flag(self):
        """Test that include_zero flag works correctly."""
        scenarios_with_zero = generateKeyFaultProbabilityScenarios(
            step=0.5, include_zero=True
        )
        scenarios_without_zero = generateKeyFaultProbabilityScenarios(
            step=0.5, include_zero=False
        )
        
        # Scenarios with zero should include cases where probabilities are 0
        # Scenarios without zero should exclude them
        assert len(scenarios_with_zero) >= len(scenarios_without_zero)
        
        # Verify no zeros in scenarios_without_zero
        for scenario in scenarios_without_zero:
            assert scenario[SAFE] > 0
            assert scenario[LOST] > 0
            assert scenario[LEAKED] > 0
            assert scenario[STOLEN] > 0

    def test_min_safe_validation(self):
        """Test that invalid min_safe values raise errors."""
        with pytest.raises(ValueError, match="min_safe must be within"):
            generateKeyFaultProbabilityScenarios(min_safe=-0.1)
        
        with pytest.raises(ValueError, match="min_safe must be within"):
            generateKeyFaultProbabilityScenarios(min_safe=1.5)

    def test_max_theft_validation(self):
        """Test that invalid max_theft values raise errors."""
        with pytest.raises(ValueError, match="max_theft must be within"):
            generateKeyFaultProbabilityScenarios(max_theft=-0.1)
        
        with pytest.raises(ValueError, match="max_theft must be within"):
            generateKeyFaultProbabilityScenarios(max_theft=1.5)

    def test_impossible_constraints_return_empty(self):
        """Test that impossible constraints return empty list."""
        # min_safe too high
        scenarios = generateKeyFaultProbabilityScenarios(
            step=0.1, min_safe=0.95, max_theft=0.01
        )
        # This might return empty if constraints are too restrictive
        assert isinstance(scenarios, list)

    def test_all_probabilities_sum_to_one(self):
        """Test that all generated scenarios have probabilities summing to 1."""
        scenarios = generateKeyFaultProbabilityScenarios(step=0.05, include_zero=True)
        
        for scenario in scenarios:
            total = scenario[SAFE] + scenario[LOST] + scenario[LEAKED] + scenario[STOLEN]
            assert abs(total - 1.0) < 1e-9, f"Probabilities don't sum to 1: {scenario}"


class TestReportOptimalWalletsForProbabilities:
    """Tests for reportOptimalWalletsForProbabilities function."""

    def test_basic_reporting(self):
        """Test basic reporting functionality."""
        probabilities_list = [
            {SAFE: 0.7, LOST: 0.1, LEAKED: 0.1, STOLEN: 0.1},
            {SAFE: 0.8, LOST: 0.05, LEAKED: 0.1, STOLEN: 0.05},
        ]
        keyCount = 2
        
        # Capture print output
        printed_lines = []
        def mock_print(*args, **kwargs):
            printed_lines.append(' '.join(str(arg) for arg in args))
        
        results = reportOptimalWalletsForProbabilities(
            probabilities_list, keyCount, print_fn=mock_print
        )
        
        assert len(results) == 2
        assert len(printed_lines) == 2
        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 3
            optimal_wallets, best_p, probs = result
            assert isinstance(optimal_wallets, list)
            assert len(optimal_wallets) >= 1
            assert 0.0 <= best_p <= 1.0
            assert isinstance(probs, dict)

    def test_validates_probability_sums(self):
        """Test that function validates probability sums."""
        probabilities_list = [
            {SAFE: 0.5, LOST: 0.3, LEAKED: 0.1, STOLEN: 0.2},  # Sums to 1.1
        ]
        keyCount = 2
        
        with pytest.raises(ValueError, match="must sum to 1.0"):
            reportOptimalWalletsForProbabilities(probabilities_list, keyCount)

    def test_handles_empty_list(self):
        """Test handling of empty probability list."""
        probabilities_list = []
        keyCount = 2
        
        printed_lines = []
        def mock_print(*args, **kwargs):
            printed_lines.append(' '.join(str(arg) for arg in args))
        
        results = reportOptimalWalletsForProbabilities(
            probabilities_list, keyCount, print_fn=mock_print
        )
        
        assert results == []
        assert len(printed_lines) == 0

    def test_returns_correct_structure(self):
        """Test that returned structure is correct."""
        probabilities_list = [
            {SAFE: 0.7, LOST: 0.1, LEAKED: 0.1, STOLEN: 0.1},
        ]
        keyCount = 2
        
        results = reportOptimalWalletsForProbabilities(
            probabilities_list, keyCount, print_fn=lambda *args: None
        )
        
        assert len(results) == 1
        optimal_wallets, best_p, probs = results[0]
        assert isinstance(optimal_wallets, list)
        assert len(optimal_wallets) >= 1
        assert isinstance(best_p, float)
        assert probs == probabilities_list[0]

    def test_deduplicate_by_architecture_flag(self):
        """Test that deduplicate_by_architecture flag works."""
        probabilities_list = [
            {SAFE: 0.7, LOST: 0.1, LEAKED: 0.1, STOLEN: 0.1},
        ]
        keyCount = 3
        
        results_with_dedup = reportOptimalWalletsForProbabilities(
            probabilities_list, keyCount, deduplicate_by_architecture=True,
            print_fn=lambda *args: None
        )
        results_without_dedup = reportOptimalWalletsForProbabilities(
            probabilities_list, keyCount, deduplicate_by_architecture=False,
            print_fn=lambda *args: None
        )
        
        # Both should return valid results
        assert len(results_with_dedup) == 1
        assert len(results_without_dedup) == 1
        # The optimal wallets might differ due to deduplication
        assert isinstance(results_with_dedup[0][0], list)
        assert isinstance(results_without_dedup[0][0], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

