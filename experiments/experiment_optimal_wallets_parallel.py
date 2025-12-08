import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from functools import partial

from helpers.computations import findOptimalWallet
from helpers.wallet_enumerations import walletStrAscii


def _process_scenario_batch(args):
    """Worker function to process a batch of probability scenarios.
    
    This function is designed to be picklable for multiprocessing.
    """
    batch_probs, wallets, keyCount, batch_start_idx = args
    
    results = []
    for local_idx, probs in enumerate(batch_probs):
        global_idx = batch_start_idx + local_idx
        optimal_wallets, _best_p = findOptimalWallet(wallets, keyCount, probs)
        is_tie = len(optimal_wallets) > 1
        
        # Convert wallets to tuples for consistent hashing
        optimal_tuples = [tuple(sorted(w)) for w in optimal_wallets]
        
        results.append({
            'index': global_idx,
            'optimal_tuples': optimal_tuples,
            'is_tie': is_tie,
        })
    
    return results


def count_optimal_wallet_occurrences_parallel(
        probabilities_list,
        wallets,
        keyCount,
        output_csv_path=None,
        print_results=True,
        num_workers=None,
        batch_size=100,
):
    """Parallel CPU version of count_optimal_wallet_occurrences.
    
    Args:
        probabilities_list: list of probability dicts
        wallets: list of wallets to track
        keyCount: number of keys in the system
        output_csv_path: optional path to save CSV
        print_results: if True, print results
        num_workers: number of worker processes (None = auto-detect CPU count)
        batch_size: number of scenarios to process per worker batch
    
    Returns:
        dict mapping wallet to occurrence counts
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"Using {num_workers} CPU cores for parallel processing")
    print(f"Batch size: {batch_size} scenarios per worker")
    
    # Convert wallets to tuples for hashing
    wallet_to_tuple = {tuple(sorted(w)): w for w in wallets}
    wallet_unique_counts = {tuple(sorted(w)): 0 for w in wallets}
    wallet_tie_counts = {tuple(sorted(w)): 0 for w in wallets}
    
    total_scenarios = len(probabilities_list)
    unique_scenarios = 0
    tie_scenarios = 0
    
    # Split scenarios into batches
    batches = []
    for batch_start in range(0, total_scenarios, batch_size):
        batch_end = min(batch_start + batch_size, total_scenarios)
        batch_probs = probabilities_list[batch_start:batch_end]
        batches.append((batch_probs, wallets, keyCount, batch_start))
    
    print(f"Processing {total_scenarios} scenarios in {len(batches)} batches...")
    
    # Process batches in parallel
    completed = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(_process_scenario_batch, batch): i 
            for i, batch in enumerate(batches)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_results = future.result()
                
                # Aggregate results
                for result in batch_results:
                    is_tie = result['is_tie']
                    if is_tie:
                        tie_scenarios += 1
                    else:
                        unique_scenarios += 1
                    
                    for optimal_tuple in result['optimal_tuples']:
                        if optimal_tuple in wallet_unique_counts:
                            if is_tie:
                                wallet_tie_counts[optimal_tuple] += 1
                            else:
                                wallet_unique_counts[optimal_tuple] += 1
                
                completed += 1
                if completed % max(1, len(batches) // 10) == 0 or completed == len(batches):
                    print(f"  Completed {completed}/{len(batches)} batches "
                          f"({completed * batch_size}/{total_scenarios} scenarios)")
                    
            except Exception as exc:
                print(f"Batch {batch_idx} generated an exception: {exc}")
                raise
    
    # Prepare results for output (same as original function)
    results = []
    for wallet_tuple in wallet_unique_counts.keys():
        unique_count = wallet_unique_counts[wallet_tuple]
        tie_count = wallet_tie_counts[wallet_tuple]
        total_count = unique_count + tie_count
        
        wallet = wallet_to_tuple[wallet_tuple]
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
    
    # Sort by unique count
    results.sort(key=lambda x: (x["Unique_Count"], x["Total_Count"]), reverse=True)
    
    if print_results:
        print(f"\nOptimal wallet occurrences across {total_scenarios} scenarios:")
        print(f"  Unique optimal: {unique_scenarios} scenarios")
        print(f"  Tied optimal: {tie_scenarios} scenarios")
        print("-" * 100)
        print(
            f"{'Wallet':<40s} | {'Unique':<8s} | {'Tied':<8s} | {'Total':<8s} | "
            f"{'Unique %':<10s} | {'Tied %':<10s} | {'Total %':<10s}"
        )
        print("-" * 100)
        for result in results:
            if result["Total_Count"] > 0:
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
        fieldnames = [
            "Wallet", "Unique_Count", "Tie_Count", "Total_Count",
            "Unique_Percentage", "Tie_Percentage", "Total_Percentage"
        ]
        with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved results to {output_csv_path}")
    
    return {
        wallet_tuple: {
            "unique_count": wallet_unique_counts[wallet_tuple],
            "tie_count": wallet_tie_counts[wallet_tuple],
            "total_count": wallet_unique_counts[wallet_tuple] + wallet_tie_counts[wallet_tuple],
        }
        for wallet_tuple in wallet_unique_counts.keys()
    }

