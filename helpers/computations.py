from helpers.wallet_enumerations import SAFE, LOST, LEAKED, STOLEN
from helpers.wallet_enumerations import enumerateStates, ownerAdvKeysFromStates, isCovered, enumerateStaticWallets, walletStr
import math


def computeSuccessProbability(wallet, ownerStates, advStates, probabilities):
    """Compute success probability for a wallet given pre-computed states and probabilities.

    Args:
        wallet: wallet to evaluate
        ownerStates: list of owner key combinations for each state
        advStates: list of adversary key combinations for each state
        probabilities: list of probabilities for each state

    Success is defined as: owner can access (covers at least one combination)
    AND adversary cannot access (covers none of the combinations).
    """
    total = 0.0
    for i in range(len(probabilities)):
        owner_ok = isCovered(ownerStates[i], wallet)
        adv_ok = isCovered(advStates[i], wallet)
        if owner_ok and not adv_ok:
            total += probabilities[i]
    return total


def findOptimalWallet(wallets, keyCount, keyStateProbabilities):
    """Return (best_wallets, best_success_probability).

    Args:
        wallets: list of wallets to evaluate
        keyCount: number of keys in the system
        keyStateProbabilities: dict with SAFE/LOST/LEAKED/STOLEN probabilities

    Returns:
        Tuple of (best_wallets, best_success_probability) where best_wallets is a list
        of all wallets with the best success probability (may contain one or more wallets).
    """
    # Compute states once for all wallets
    states, state_probabilities = enumerateStates(keyCount, keyStateProbabilities)
    ownerStates, advStates = ownerAdvKeysFromStates(states)
    
    best_wallets = []
    best_prob = -1.0
    for wallet in wallets:
        p = computeSuccessProbability(wallet, ownerStates, advStates, state_probabilities)
        if abs(p - best_prob) < 1e-12:  # Equal probability (within floating point tolerance)
            best_wallets.append(wallet)
        elif p > best_prob:
            best_prob = p
            best_wallets = [wallet]  # Start new list with this wallet

    return best_wallets, best_prob


def generateKeyFaultProbabilityScenarios(step=0.05, include_zero=False, min_safe=0.5, max_theft=0.5):
    """Generate probability scenarios on an exact integer grid that sum to 1.

    - step: grid granularity (e.g., 0.5, 0.25, 0.2, 0.1)
    - include_zero: if False, excludes scenarios where any probability is 0.0
    - min_safe: if provided, enforces SAFE probability >= this threshold
    - max_theft: if provided, enforces STOLEN probability <= this threshold
    """
    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1]")
    if min_safe is not None and (min_safe < 0.0 or min_safe > 1.0):
        raise ValueError("min_safe must be within [0, 1]")
    if max_theft is not None and (max_theft < 0.0 or max_theft > 1.0):
        raise ValueError("max_theft must be within [0, 1]")

    # Use integer grid to avoid floating drift: a+b+c+d = n, probabilities = a/n, ...
    n_float = 1.0 / step
    n = int(round(n_float))
    if abs(n - n_float) > 1e-9:
        # Guard: step must evenly divide 1.0 for an exact grid
        raise ValueError("step must evenly divide 1.0 (e.g., 0.5, 0.25, 0.2, 0.1)")

    scenarios = []
    # Compute minimal integer count for SAFE given min_safe
    if min_safe is None:
        min_a = 0
    else:
        # Align threshold to grid: a/n >= min_safe  ->  a >= ceil(min_safe * n)
        min_a = int(math.ceil(min_safe * n - 1e-12))
        if min_a > n:
            return []
    
    # Compute maximal integer count for STOLEN given max_theft
    if max_theft is None:
        max_d = n
    else:
        # Align threshold to grid: d/n <= max_theft  ->  d <= floor(max_theft * n)
        max_d = int(math.floor(max_theft * n + 1e-12))
        if max_d < 0:
            return []

    for a in range(min_a, n + 1):  # SAFE
        for b in range(n - a + 1):  # LOST
            for c in range(n - a - b + 1):  # LEAKED
                d = n - (a + b + c)  # STOLEN
                
                # Skip if STOLEN exceeds max_theft
                if d > max_d:
                    continue
                
                # Convert to exact floats via division by n
                p_safe = a / n
                p_lost = b / n
                p_leaked = c / n
                p_stolen = d / n

                if not include_zero and (
                    p_safe == 0.0 or p_lost == 0.0 or p_leaked == 0.0 or p_stolen == 0.0
                ):
                    continue

                scenarios.append(
                    {
                        SAFE: p_safe,
                        LOST: p_lost,
                        LEAKED: p_leaked,
                        STOLEN: p_stolen,
                    }
                )
    return scenarios


def reportOptimalWalletsForProbabilities(
    probabilities_list,
    keyCount,
    deduplicate_by_architecture=True,
    print_fn=print,
):
    """For each probabilities dict in the list, print and return the best wallet.

    Args:
        probabilities_list: iterable of dicts keyed by SAFE/LOST/LEAKED/STOLEN.
        keyCount: number of keys in the system.
        deduplicate_by_architecture: whether to deduplicate wallets by architecture.
        print_fn: callable used for printing (default: built-in print).

    Returns:
        List of tuples: (wallet, best_success_probability, probabilities_dict)
    """
    # Generate wallets once for all probability scenarios
    wallets = enumerateStaticWallets(
        keyCount, deduplicate_by_architecture=deduplicate_by_architecture
    )
    
    results = []
    for idx, probs in enumerate(probabilities_list):
        # Basic validation: ensure probabilities sum to ~1
        total = (
            probs.get(SAFE, 0.0)
            + probs.get(LOST, 0.0)
            + probs.get(LEAKED, 0.0)
            + probs.get(STOLEN, 0.0)
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"probabilities at index {idx} must sum to 1.0 (got {total})"
            )

        optimal_wallets, best_p = findOptimalWallet(wallets, keyCount, probs)
        # Format multiple wallets if there are ties
        if len(optimal_wallets) == 1:
            wallet_str = walletStr(optimal_wallets[0])
        else:
            wallet_strs = [walletStr(w) for w in optimal_wallets]
            wallet_str = f"[{len(optimal_wallets)} wallets: {', '.join(wallet_strs)}]"
        print_fn(
            f"Case {idx}: best_success_probability={best_p:.6f}, wallet(s)={wallet_str}, probs={probs}"
        )
        # Return the first wallet for backward compatibility, or all wallets
        results.append((optimal_wallets, best_p, probs))
    return results
