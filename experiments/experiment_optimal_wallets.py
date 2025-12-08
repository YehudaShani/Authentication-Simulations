import csv

from helpers.computations import (
    generateKeyFaultProbabilityScenarios,
    reportOptimalWalletsForProbabilities,
    findOptimalWallet,
    computeSuccessProbability,
)
from helpers.wallet_enumerations import (
    walletStrAscii,
    SAFE,
    LOST,
    LEAKED,
    STOLEN,
    enumerateStaticWallets,
    enumerateStates,
    ownerAdvKeysFromStates,
)


def check_optimal_wallets_over_scenarios():
    print("Evaluating optimal wallets over a few scenarios...")
    scenarios = generateKeyFaultProbabilityScenarios(step=0.05, include_zero=False, min_safe=0.5)
    sample = scenarios[:15]  # take a small sample for demonstration
    reportOptimalWalletsForProbabilities(sample, keyCount=4)


def find_optimal_wallets_for_different_key_counts(
        output_csv_path,
        key_counts,
        step=0.05,
        include_zero=False,
        min_safe=0.1,
        sample_size=None,
):
    """Generate scenarios and save pivot CSV: rows=scenarios, cols=keyCounts, cells=wallets.

    The first column is a human-readable scenario label; each subsequent column is the
    optimal wallet (repr) for that keyCount under the scenario.
    """
    scenarios = generateKeyFaultProbabilityScenarios(step=step, include_zero=include_zero, min_safe=min_safe)
    if sample_size is not None:
        scenarios = scenarios[:sample_size]

    def scenario_label(p):
        # Use canonical keys from helpers to avoid key mismatches
        p_safe = p.get(SAFE, 0.0)
        p_lost = p.get(LOST, 0.0)
        p_leaked = p.get(LEAKED, 0.0)
        p_stolen = p.get(STOLEN, 0.0)
        return f"SAFE={p_safe:.4f},LOST={p_lost:.4f},LEAKED={p_leaked:.4f},STOLEN={p_stolen:.4f}"

    # Generate wallets for each keyCount once (more efficient)
    wallets_by_keycount = {
        k: enumerateStaticWallets(k, deduplicate_by_architecture=True)
        for k in key_counts
    }

    # Header: Scenario + columns for each keyCount
    header = ["Scenario"] + [str(k) for k in key_counts]
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for probs in scenarios:
            row = [scenario_label(probs)]
            for k in key_counts:
                wallets = wallets_by_keycount[k]
                optimal_wallets, best_p = findOptimalWallet(wallets, k, probs)
                # Format multiple wallets if there are ties
                if len(optimal_wallets) == 1:
                    wallet_str = walletStrAscii(optimal_wallets[0])
                else:
                    wallet_strs = [walletStrAscii(w) for w in optimal_wallets]
                    wallet_str = f"[TIE: {len(optimal_wallets)} wallets: {' | '.join(wallet_strs)}]"
                row.append(f"{wallet_str} (p={best_p:.6f})")
            writer.writerow(row)
    print(f"Wrote CSV with {len(scenarios)} scenarios and {len(key_counts)} keyCounts to {output_csv_path}")


def rank_wallets_by_success_probability(
        probabilities_list,
        wallets,
        keyCount,
        output_csv_path,
):
    """Rank wallets by success probability for each probability scenario and save to CSV.

    Args:
        probabilities_list: list of probability dicts (each with SAFE/LOST/LEAKED/STOLEN keys)
        wallets: list of wallets to evaluate
        keyCount: number of keys in the system
        output_csv_path: path to output CSV file

    Output CSV format:
        - Scenario: probability scenario label
        - Rank: ranking (1 = best)
        - Wallet: wallet string representation
        - Success_Probability: success probability for this wallet under this scenario
    """

    def scenario_label(p):
        p_safe = p.get(SAFE, 0.0)
        p_lost = p.get(LOST, 0.0)
        p_leaked = p.get(LEAKED, 0.0)
        p_stolen = p.get(STOLEN, 0.0)
        return f"SAFE={p_safe:.4f},LOST={p_lost:.4f},LEAKED={p_leaked:.4f},STOLEN={p_stolen:.4f}"

    rows = []
    for probs in probabilities_list:
        # Compute states once for this probability scenario
        states, state_probabilities = enumerateStates(keyCount, probs)
        ownerStates, advStates = ownerAdvKeysFromStates(states)

        # Compute success probability for each wallet under this scenario
        wallet_scores = []
        for wallet in wallets:
            p = computeSuccessProbability(wallet, ownerStates, advStates, state_probabilities)
            wallet_scores.append((wallet, p))

        # Sort by success probability (descending) and assign ranks
        wallet_scores.sort(key=lambda x: x[1], reverse=True)

        # Handle ties: same rank for same probability
        current_rank = 1
        prev_prob = None
        for idx, (wallet, prob) in enumerate(wallet_scores):
            if prev_prob is not None and abs(prob - prev_prob) > 1e-12:
                current_rank = idx + 1
            prev_prob = prob
            rows.append({
                "Scenario": scenario_label(probs),
                "Rank": current_rank,
                "Wallet": walletStrAscii(wallet),
                "Success_Probability": f"{prob:.6f}",
            })

    fieldnames = ["Scenario", "Rank", "Wallet", "Success_Probability"]
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} ranked wallet entries to {output_csv_path}")


def count_optimal_wallet_occurrences(
        probabilities_list,
        wallets,
        keyCount,
        output_csv_path=None,
        print_results=True,
):
    """Count how many times each wallet was optimal across probability scenarios.

    Tracks both unique optimal occurrences and tied optimal occurrences separately.

    Args:
        probabilities_list: list of probability dicts (each with SAFE/LOST/LEAKED/STOLEN keys)
        wallets: list of wallets to track (these are the wallets we're counting)
        keyCount: number of keys in the system
        output_csv_path: optional path to save CSV file with results
        print_results: if True, print results to console

    Returns:
        dict mapping wallet (as tuple for hashing) to dict with 'unique_count' and 'tie_count'
    """
    # Convert wallets to tuples for hashing and create a mapping (sort for consistent comparison)
    wallet_to_tuple = {tuple(sorted(w)): w for w in wallets}
    wallet_unique_counts = {tuple(sorted(w)): 0 for w in wallets}
    wallet_tie_counts = {tuple(sorted(w)): 0 for w in wallets}

    total_scenarios = 0
    unique_scenarios = 0  # Scenarios with unique optimal wallet
    tie_scenarios = 0  # Scenarios with tied optimal wallets

    for probs in probabilities_list:
        total_scenarios += 1
        # Find optimal wallets for this probability scenario from the provided wallets
        optimal_wallets, _best_p = findOptimalWallet(wallets, keyCount, probs)

        is_tie = len(optimal_wallets) > 1
        if is_tie:
            tie_scenarios += 1
        else:
            unique_scenarios += 1

        # Count optimal wallets (track unique vs tied separately)
        for optimal_wallet in optimal_wallets:
            optimal_tuple = tuple(sorted(optimal_wallet))  # Sort for consistent comparison
            # Check if this optimal wallet is in our tracked wallets
            if optimal_tuple in wallet_unique_counts:
                if is_tie:
                    wallet_tie_counts[optimal_tuple] += 1
                else:
                    wallet_unique_counts[optimal_tuple] += 1

    # Prepare results for output
    results = []
    for wallet_tuple in wallet_unique_counts.keys():
        unique_count = wallet_unique_counts[wallet_tuple]
        tie_count = wallet_tie_counts[wallet_tuple]
        total_count = unique_count + tie_count

        wallet = wallet_to_tuple[wallet_tuple]
        # Calculate percentages based on total scenarios
        unique_percentage = (unique_count / unique_scenarios * 100) if unique_scenarios > 0 else 0.0
        tie_percentage = (tie_count / tie_scenarios * 100) if tie_scenarios > 0 else 0.0
        total_percentage = (total_count / total_scenarios * 100) if total_scenarios > 0 else 0.0

        results.append({
            "Wallet": walletStrAscii(wallet),
            "Unique_Count": unique_count,
            "Tie_Count": tie_count,
            "Total_Count": total_count,
            "Unique_Percentage": f"{unique_percentage:.2f}%",
            "Tie_Percentage": f"{tie_percentage:.2f}%",
            "Total_Percentage": f"{total_percentage:.2f}%",
        })

    # Sort by unique count (descending), then by total count as tiebreaker
    results.sort(key=lambda x: (x["Unique_Count"], x["Total_Count"]), reverse=True)

    if print_results:
        print(f"\nOptimal wallet occurrences across {total_scenarios} scenarios:")
        print(f"  Unique optimal: {unique_scenarios} scenarios")
        print(f"  Tied optimal: {tie_scenarios} scenarios")
        print("-" * 100)
        print(
            f"{'Wallet':<40s} | {'Unique':<8s} | {'Tied':<8s} | {'Total':<8s} | {'Unique %':<10s} | {'Tied %':<10s} | {'Total %':<10s}")
        print("-" * 100)
        for result in results:
            if result["Total_Count"] > 0:  # Only show wallets that were optimal at least once
                print(
                    f"{result['Wallet']:<40s} | "
                    f"{result['Unique_Count']:>6d}   | "
                    f"{result['Tie_Count']:>6d}   | "
                    f"{result['Total_Count']:>6d}   | "
                    f"{result['Unique_Percentage']:>9s} | "
                    f"{result['Tie_Percentage']:>9s} | "
                    f"{result['Total_Percentage']:>9s}"
                )

    if output_csv_path:
        fieldnames = ["Wallet", "Unique_Count", "Tie_Count", "Total_Count", "Unique_Percentage", "Tie_Percentage",
                      "Total_Percentage"]
        with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved results to {output_csv_path}")

    # Return both counts for programmatic access
    return {
        wallet_tuple: {
            "unique_count": wallet_unique_counts[wallet_tuple],
            "tie_count": wallet_tie_counts[wallet_tuple],
            "total_count": wallet_unique_counts[wallet_tuple] + wallet_tie_counts[wallet_tuple],
        }
        for wallet_tuple in wallet_unique_counts.keys()
    }


def analyze_wallet_transitions(
    keyCount_from,
    keyCount_to,
    step=0.05,
    include_zero=False,
    min_safe=0.1,
    sample_size=None,
    deduplicate_by_architecture=True,
    output_csv_path=None,
):
    """
    For each probability scenario, find optimal wallets for keyCount_from (n keys)
    and keyCount_to (typically n+1 keys), and count how often each 'from' wallet
    transitions to each 'to' wallet.

    Args:
        keyCount_from: smaller key count (n).
        keyCount_to: larger key count (e.g., n+1).
        step, include_zero, min_safe: parameters for generateKeyFaultProbabilityScenarios.
        sample_size: optional limit on the number of scenarios to evaluate.
        deduplicate_by_architecture: whether to deduplicate wallets by architecture.
        output_csv_path: if provided, write a CSV summarizing transitions.

    Returns:
        transitions: dict keyed by (from_wallet_tuple, to_wallet_tuple) with integer 'count'.
        from_totals: dict keyed by from_wallet_tuple with total outgoing transitions.
        is_unique: dict keyed by (from_wallet_tuple, to_wallet_tuple) with boolean indicating if transition is ever unique.
    """
    assert keyCount_to > keyCount_from, "keyCount_to should be > keyCount_from"

    # 1) Generate probability scenarios
    scenarios = generateKeyFaultProbabilityScenarios(
        step=step,
        include_zero=include_zero,
        min_safe=min_safe,
    )
    if sample_size is not None:
        scenarios = scenarios[:sample_size]

    # 2) Enumerate wallets for both key counts
    wallets_from = enumerateStaticWallets(
        keyCount_from, deduplicate_by_architecture=deduplicate_by_architecture
    )
    wallets_to = enumerateStaticWallets(
        keyCount_to, deduplicate_by_architecture=deduplicate_by_architecture
    )

    # Transition counts:
    # (tuple(sorted(from_wallet)), tuple(sorted(to_wallet))) -> count
    transitions = {}
    from_totals = {}  # total outgoing transitions from each from_wallet
    is_unique = {}  # track if transition ever occurred with no ties on either side

    for probs in scenarios:
        # Optimal wallets for n keys
        opt_from, _ = findOptimalWallet(wallets_from, keyCount_from, probs)
        # Optimal wallets for keyCount_to keys
        opt_to, _ = findOptimalWallet(wallets_to, keyCount_to, probs)

        # Check if this scenario has unique optima on both sides
        from_is_unique = len(opt_from) == 1
        to_is_unique = len(opt_to) == 1
        scenario_is_unique = from_is_unique and to_is_unique

        # Record all pairwise transitions (including ties)
        for wf in opt_from:
            wf_key = tuple(sorted(wf))
            from_totals.setdefault(wf_key, 0)

            for wt in opt_to:
                wt_key = tuple(sorted(wt))
                key = (wf_key, wt_key)
                transitions[key] = transitions.get(key, 0) + 1
                from_totals[wf_key] += 1
                
                # Mark as unique if this scenario had no ties on either side
                if scenario_is_unique:
                    is_unique[key] = True
                elif key not in is_unique:
                    # Initialize to False if not already set to True
                    is_unique[key] = False

    # Add all possible transitions that never occurred (with count 0)
    for wf in wallets_from:
        wf_key = tuple(sorted(wf))
        for wt in wallets_to:
            wt_key = tuple(sorted(wt))
            key = (wf_key, wt_key)
            if key not in transitions:
                transitions[key] = 0
                is_unique[key] = False
                # Ensure from_wallet is in from_totals (even if 0)
                if wf_key not in from_totals:
                    from_totals[wf_key] = 0

    # Optionally write to CSV
    if output_csv_path:
        with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "From_keyCount",
                    "To_keyCount",
                    "From_wallet",
                    "To_wallet",
                    "Transition_count",
                    "From_total",
                    "Transition_fraction",
                    "Is_unique",
                ]
            )
            # Sort transitions by from_wallet, then by to_wallet
            sorted_transitions = sorted(transitions.items(), key=lambda x: (x[0][0], x[0][1]))
            
            for (wf_key, wt_key), count in sorted_transitions:
                from_total = from_totals.get(wf_key, 0)
                frac = count / from_total if from_total > 0 else 0.0
                wf = list(wf_key)
                wt = list(wt_key)
                unique_flag = is_unique.get((wf_key, wt_key), False)
                writer.writerow(
                    [
                        keyCount_from,
                        keyCount_to,
                        walletStrAscii(wf),
                        walletStrAscii(wt),
                        count,
                        from_total,
                        f"{frac:.6f}",
                        unique_flag,
                    ]
                )

    return transitions, from_totals, is_unique


if __name__ == "__main__":
    # Example usage: analyze how optimal wallets transition from n to n+1 keys
    step = 0.02
    minsafe = 0.2

    keyCount_from = 2
    keyCount_to = 3

    print(
        f"Analyzing wallet transitions from {keyCount_from} to {keyCount_to} "
        f"keys (step={step}, min_safe={minsafe})"
    )

    transitions, from_totals, is_unique = analyze_wallet_transitions(
        keyCount_from=keyCount_from,
        keyCount_to=keyCount_to,
        step=step,
        include_zero=False,
        min_safe=minsafe,
        sample_size=None,  # or set to an int for a quick sample
        deduplicate_by_architecture=True,
        output_csv_path=f"wallet_transitions_{keyCount_from}to{keyCount_to}_step_{step}_minsafe_{minsafe}2.csv",
    )

    print(f"Recorded {len(transitions)} distinct transitions.")
