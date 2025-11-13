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


if __name__ == "__main__":
    step = 0.02
    minsafe = 0.2

    probabilities = [{SAFE: 0.3, LOST: 0.26, LEAKED: 0.17, STOLEN: 0.27}]
    print("Generated", len(probabilities), "probability scenarios.")

    keyCount = 4
    wallets = enumerateStaticWallets(keyCount, deduplicate_by_architecture=True)
    print("Generated", len(wallets), "wallets for keyCount =", keyCount)

    rank_wallets_by_success_probability(
        probabilities_list=probabilities,
        wallets=wallets,
        keyCount=keyCount,
        output_csv_path=f"rank_by_probabilty_{keyCount}_keys_step_{step}_minsafe_{minsafe}_ties_separated.csv",
    )
