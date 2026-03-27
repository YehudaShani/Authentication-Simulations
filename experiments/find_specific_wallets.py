import csv
import json

from helpers.computations import findOptimalWallet
from helpers.wallet_enumerations import walletStrAscii, enumerateStaticWallets
from consts import SAFE, LOST, LEAKED, STOLEN


def find_optimal_probability_for_each_wallet(
    probabilities_list,
    wallets,
    keyCount,
    output_csv_path=None,
):
    """Find one probability scenario where each wallet is optimal.
    
    For each wallet in the provided list, finds the first probability scenario
    from probabilities_list where that wallet is optimal.
    
    Args:
        probabilities_list: list of probability dicts (each with SAFE/LOST/LEAKED/STOLEN)
        wallets: list of wallets to check
        keyCount: number of keys in the system
        output_csv_path: optional path to save results as CSV
        output_json_path: optional path to save results as JSON
    
    Returns:
        dict mapping wallet (as string) to probability dict where it's optimal.
        Wallets that are never optimal will not be in the dict.
    """
    # Convert wallets to tuples for comparison
    wallet_to_str = {tuple(sorted(w)): walletStrAscii(w) for w in wallets}
    wallet_to_obj = {tuple(sorted(w)): w for w in wallets}
    
    # Track which wallets we've found optimal scenarios for
    wallet_optimal_probs = {}
    
    print(f"Searching for optimal probability scenarios for {len(wallets)} wallets...")
    print(f"Checking {len(probabilities_list)} probability scenarios...")
    
    for idx, probs in enumerate(probabilities_list):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(probabilities_list)} scenarios, "
                  f"found {len(wallet_optimal_probs)}/{len(wallets)} wallets...")
        
        # Find optimal wallets for this probability scenario
        optimal_wallets, _ = findOptimalWallet(wallets, keyCount, probs)
        
        # Check each optimal wallet
        for optimal_wallet in optimal_wallets:
            wallet_tuple = tuple(sorted(optimal_wallet))
            
            # If we haven't found a scenario for this wallet yet, save it
            if wallet_tuple not in wallet_optimal_probs:
                wallet_optimal_probs[wallet_tuple] = probs.copy()
                
                # Early exit if we've found scenarios for all wallets
                if len(wallet_optimal_probs) == len(wallets):
                    print(f"  Found optimal scenarios for all wallets!")
                    break
        
        # Early exit if we've found scenarios for all wallets
        if len(wallet_optimal_probs) == len(wallets):
            break
    
    print(f"\nCompleted: Found optimal scenarios for {len(wallet_optimal_probs)}/{len(wallets)} wallets")
    
    # Convert to string-based dict for output
    result = {}
    for wallet_tuple, probs in wallet_optimal_probs.items():
        wallet_str = wallet_to_str[wallet_tuple]
        result[wallet_str] = probs
    
    # Save to CSV if requested
    if output_csv_path:
        with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Wallet", "SAFE", "LOST", "LEAKED", "STOLEN"])
            for wallet_str, probs in sorted(result.items()):
                writer.writerow([
                    wallet_str,
                    probs.get(SAFE, 0.0),
                    probs.get(LOST, 0.0),
                    probs.get(LEAKED, 0.0),
                    probs.get(STOLEN, 0.0),
                ])
        print(f"Saved results to {output_csv_path}")

    
    return result



def main():
    """Main function to find optimal probability scenarios for each wallet."""
    from helpers.computations import generateKeyFaultProbabilityScenarios
    
    # Configuration
    keyCount = 4
    step = 0.05
    min_safe = 0.3
    max_theft = 0.4
    deduplicate_by_architecture = True
    
    print("=" * 80)
    print(f"Finding optimal probability scenarios for keyCount={keyCount}")
    print("=" * 80)
    
    # Generate wallets
    wallets = enumerateStaticWallets(keyCount, deduplicate_by_architecture=deduplicate_by_architecture)
    print(f"Generated {len(wallets)} wallets for keyCount={keyCount}")
    
    # Generate probability scenarios
    probabilities_list = generateKeyFaultProbabilityScenarios(
        step=step,
        include_zero=False,
        min_safe=min_safe,
        max_theft=max_theft
    )
    print(f"Generated {len(probabilities_list)} probability scenarios")
    print(f"  step={step}, min_safe={min_safe}, max_theft={max_theft}")
    
    # Find optimal probability for each wallet
    result = find_optimal_probability_for_each_wallet(
        probabilities_list=probabilities_list,
        wallets=wallets,
        keyCount=keyCount,
        output_csv_path=f"wallet_optimal_probabilities_keyCount_{keyCount}.csv",
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  Total wallets checked: {len(wallets)}")
    print(f"  Wallets with optimal scenarios found: {len(result)}")
    print(f"  Wallets never optimal: {len(wallets) - len(result)}")
    
    if result:
        print(f"\nFirst 5 results:")
        for i, (wallet, probs) in enumerate(list(result.items())[:5], 1):
            print(f"  {i}. {wallet}")
            print(f"     SAFE={probs[SAFE]:.4f}, LOST={probs[LOST]:.4f}, "
                  f"LEAKED={probs[LEAKED]:.4f}, STOLEN={probs[STOLEN]:.4f}")


if __name__ == "__main__":
    main()

