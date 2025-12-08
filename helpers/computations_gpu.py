"""
GPU-accelerated computations for wallet optimization.

This module provides GPU-accelerated versions of the core computation functions
using CuPy (CUDA) or NumPy (CPU fallback).
"""

import numpy as np

# Try to import CuPy for GPU acceleration, fallback to NumPy
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy detected - GPU acceleration available")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("CuPy not available - using CPU (NumPy)")

from helpers.wallet_enumerations import SAFE, LOST, LEAKED, STOLEN
from helpers.wallet_enumerations import enumerateStates, ownerAdvKeysFromStates


def is_covered_vectorized(key_combinations, wallet_combinations, xp=np):
    """
    Vectorized version of isCovered that checks multiple key combinations
    against wallet combinations using GPU/CPU arrays.
    
    Args:
        key_combinations: array of key combination bitmasks (shape: [n_keys])
        wallet_combinations: array of wallet combination bitmasks (shape: [n_wallet_combs])
        xp: array module (numpy or cupy)
    
    Returns:
        Boolean array indicating if each key combination is covered (shape: [n_keys])
    """
    # Expand dimensions for broadcasting: [n_keys, 1] & [1, n_wallet_combs]
    key_expanded = key_combinations[:, xp.newaxis]  # [n_keys, 1]
    wallet_expanded = wallet_combinations[xp.newaxis, :]  # [1, n_wallet_combs]
    
    # Bitwise AND: [n_keys, n_wallet_combs]
    and_result = key_expanded & wallet_expanded
    
    # Check if any wallet combination matches: (key & wallet == wallet)
    matches = (and_result == wallet_expanded)  # [n_keys, n_wallet_combs]
    
    # Any match means covered: [n_keys]
    return matches.any(axis=1)


def compute_success_probabilities_batch(
    wallets, owner_states, adv_states, state_probabilities, xp=np
):
    """
    Compute success probabilities for all wallets across all states in batch.
    
    Args:
        wallets: list of wallets (each wallet is a list of bitmask combinations)
        owner_states: array of owner key combinations for each state [n_states]
        adv_states: array of adversary key combinations for each state [n_states]
        state_probabilities: array of probabilities for each state [n_states]
        xp: array module (numpy or cupy)
    
    Returns:
        Array of success probabilities for each wallet [n_wallets]
    """
    n_states = len(state_probabilities)
    n_wallets = len(wallets)
    
    # Convert to arrays
    owner_states_arr = xp.asarray(owner_states, dtype=xp.int32)
    adv_states_arr = xp.asarray(adv_states, dtype=xp.int32)
    state_probs_arr = xp.asarray(state_probabilities, dtype=xp.float32)
    
    # Pre-allocate result array
    wallet_probs = xp.zeros(n_wallets, dtype=xp.float32)
    
    # Process each wallet
    for w_idx, wallet in enumerate(wallets):
        wallet_combs = xp.asarray(wallet, dtype=xp.int32)
        
        # Check coverage for all states at once
        owner_covered = is_covered_vectorized(owner_states_arr, wallet_combs, xp)
        adv_covered = is_covered_vectorized(adv_states_arr, wallet_combs, xp)
        
        # Success: owner can access AND adversary cannot
        success_mask = owner_covered & (~adv_covered)
        
        # Sum probabilities for successful states
        wallet_probs[w_idx] = xp.sum(state_probs_arr[success_mask])
    
    return wallet_probs


def find_optimal_wallets_batch(
    wallets, key_count, key_state_probabilities_list, xp=np, batch_size=None
):
    """
    Find optimal wallets for multiple probability scenarios in batch.
    
    Args:
        wallets: list of wallets to evaluate
        key_count: number of keys in the system
        key_state_probabilities_list: list of probability dicts (one per scenario)
        xp: array module (numpy or cupy)
        batch_size: number of scenarios to process at once (None = all)
    
    Returns:
        List of tuples: (best_wallets_list, best_prob_list) for each scenario
    """
    from helpers.computations import enumerateStates, ownerAdvKeysFromStates
    
    n_scenarios = len(key_state_probabilities_list)
    results = []
    
    # Process in batches to manage memory
    if batch_size is None:
        batch_size = n_scenarios
    
    for batch_start in range(0, n_scenarios, batch_size):
        batch_end = min(batch_start + batch_size, n_scenarios)
        batch_probs = key_state_probabilities_list[batch_start:batch_end]
        
        batch_results = []
        for probs in batch_probs:
            # Compute states for this probability scenario
            states, state_probabilities = enumerateStates(key_count, probs)
            owner_states, adv_states = ownerAdvKeysFromStates(states)
            
            # Compute success probabilities for all wallets
            wallet_probs = compute_success_probabilities_batch(
                wallets, owner_states, adv_states, state_probabilities, xp
            )
            
            # Convert to CPU numpy if using GPU
            if GPU_AVAILABLE and xp is cp:
                wallet_probs_cpu = cp.asnumpy(wallet_probs)
            else:
                wallet_probs_cpu = wallet_probs
            
            # Find optimal wallets (handle ties)
            best_prob = float(wallet_probs_cpu.max())
            best_indices = np.where(np.abs(wallet_probs_cpu - best_prob) < 1e-12)[0]
            best_wallets = [wallets[i] for i in best_indices]
            
            batch_results.append((best_wallets, best_prob))
        
        results.extend(batch_results)
    
    return results


def count_optimal_wallet_occurrences_gpu(
    probabilities_list,
    wallets,
    key_count,
    output_csv_path=None,
    print_results=True,
    batch_size=1000,
    use_gpu=True,
):
    """
    GPU-accelerated version of count_optimal_wallet_occurrences.
    
    Args:
        probabilities_list: list of probability dicts
        wallets: list of wallets to track
        key_count: number of keys in the system
        output_csv_path: optional path to save CSV
        print_results: if True, print results
        batch_size: number of scenarios to process per GPU batch
        use_gpu: if True, use GPU (if available), else use CPU
    
    Returns:
        dict mapping wallet to occurrence counts
    """
    import csv
    from helpers.wallet_enumerations import walletStrAscii
    
    # Select array module
    if use_gpu and GPU_AVAILABLE:
        xp = cp
        print(f"Using GPU acceleration (CuPy) with batch_size={batch_size}")
    else:
        xp = np
        print(f"Using CPU (NumPy) with batch_size={batch_size}")
    
    # Convert wallets to tuples for hashing
    wallet_to_tuple = {tuple(sorted(w)): w for w in wallets}
    wallet_unique_counts = {tuple(sorted(w)): 0 for w in wallets}
    wallet_tie_counts = {tuple(sorted(w)): 0 for w in wallets}
    
    total_scenarios = len(probabilities_list)
    unique_scenarios = 0
    tie_scenarios = 0
    
    # Process scenarios in batches
    print(f"Processing {total_scenarios} scenarios in batches of {batch_size}...")
    
    for batch_start in range(0, total_scenarios, batch_size):
        batch_end = min(batch_start + batch_size, total_scenarios)
        batch_probs = probabilities_list[batch_start:batch_end]
        
        if (batch_start // batch_size + 1) % 10 == 0 or batch_end == total_scenarios:
            print(f"  Processing batch {batch_start // batch_size + 1} "
                  f"({batch_start}-{batch_end-1} / {total_scenarios})")
        
        # Find optimal wallets for this batch
        batch_results = find_optimal_wallets_batch(
            wallets, key_count, batch_probs, xp=xp, batch_size=None
        )
        
        # Count occurrences
        for optimal_wallets, _best_p in batch_results:
            is_tie = len(optimal_wallets) > 1
            if is_tie:
                tie_scenarios += 1
            else:
                unique_scenarios += 1
            
            for optimal_wallet in optimal_wallets:
                optimal_tuple = tuple(sorted(optimal_wallet))
                if optimal_tuple in wallet_unique_counts:
                    if is_tie:
                        wallet_tie_counts[optimal_tuple] += 1
                    else:
                        wallet_unique_counts[optimal_tuple] += 1
    
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

