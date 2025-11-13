from ..helpers.consts import SAFE, LOST, LEAKED, STOLEN
from helpers.computations import (
    generateKeyFaultProbabilityScenarios,
    findOptimalWallet,
    computeSuccessProbability,
)
from helpers.wallet_enumerations import enumerateStaticWallets, enumerateStates, ownerAdvKeysFromStates




class TestWallets:
    def test_find_optimal_wallet_small(self):
        keyCount = 3
        probs = {SAFE: 0.7, LOST: 0.1, LEAKED: 0.1, STOLEN: 0.1}
        wallets = enumerateStaticWallets(keyCount, deduplicate_by_architecture=True)
        optimal_wallets, best_p = findOptimalWallet(wallets, keyCount, probs)
        assert isinstance(optimal_wallets, list)
        assert len(optimal_wallets) > 0
        assert 0.0 <= best_p <= 1.0
        # Verify that all optimal wallets have the same probability
        states, state_probabilities = enumerateStates(keyCount, probs)
        ownerStates, advStates = ownerAdvKeysFromStates(states)
        for wallet in optimal_wallets:
            assert isinstance(wallet, list)
            recomputed_p = computeSuccessProbability(wallet, ownerStates, advStates, state_probabilities)
            assert abs(best_p - recomputed_p) < 1e-9

    def test_generate_probabilities_grid(self):
        scenarios = generateKeyFaultProbabilityScenarios(step=0.5, include_zero=True)
        assert len(scenarios) > 0
        for s in scenarios:
            assert abs(s[SAFE] + s[LOST] + s[LEAKED] + s[STOLEN] - 1.0) < 1e-9


    def test_compute_success_probability_bounds(self):
        keyCount = 3
        probs = {SAFE: 0.6, LOST: 0.2, LEAKED: 0.1, STOLEN: 0.1}
        wallets = enumerateStaticWallets(keyCount, deduplicate_by_architecture=True)
        # Compute states once for all wallets
        states, state_probabilities = enumerateStates(keyCount, probs)
        ownerStates, advStates = ownerAdvKeysFromStates(states)
        for w in wallets:
            p = computeSuccessProbability(w, ownerStates, advStates, state_probabilities)
            assert 0.0 <= p <= 1.0

