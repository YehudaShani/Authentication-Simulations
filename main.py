from experiments.experiment_optimal_wallets_parallel import count_optimal_wallet_occurrences_parallel
from helpers.computations_parallel import generateKeyFaultProbabilityScenariosParallel
from helpers.wallet_enumerations import enumerateStaticWallets


def main():
    # Configuration
    keyCount = 4
    num_probabilities = 100000
    step = 0.0001  # Small step to generate many scenarios
    min_safe = 0.1
    deduplicate_by_architecture = True
    num_workers = None  # None = auto-detect CPU cores, or set to specific number
    batch_size = 1000  # Number of scenarios per worker batch
    
    print(f"Generating probability scenarios (parallel CPU)...")
    print(f"  step={step}, min_safe={min_safe}, keyCount={keyCount}")
    
    # Generate probability scenarios (parallel version)
    all_scenarios = generateKeyFaultProbabilityScenariosParallel(
        step=step,
        include_zero=False,
        min_safe=min_safe,
        num_workers=num_workers
    )
    
    # Take the first num_probabilities scenarios if needed
    if len(all_scenarios) > num_probabilities:
        probabilities_list = all_scenarios[:num_probabilities]
        print(f"Using first {len(probabilities_list)} of {len(all_scenarios)} generated scenarios")
    else:
        probabilities_list = all_scenarios
        print(f"Using all {len(probabilities_list)} generated scenarios")
    
    # Generate wallets
    print(f"\nGenerating wallets for keyCount={keyCount}...")
    wallets = enumerateStaticWallets(
        keyCount, 
        deduplicate_by_architecture=deduplicate_by_architecture
    )
    print(f"Generated {len(wallets)} wallets")
    
    # Count optimal wallet occurrences (parallel CPU version)
    print("\nCounting optimal wallet occurrences (parallel CPU)...")
    results = count_optimal_wallet_occurrences_parallel(
        probabilities_list=probabilities_list,
        wallets=wallets,
        keyCount=keyCount,
        output_csv_path="optimal_wallet_occurrences_100000.csv",
        print_results=True,
        num_workers=num_workers,
        batch_size=batch_size
    )
    
    print(f"\nAnalysis complete! Results saved to optimal_wallet_occurrences_100000.csv")


if __name__ == "__main__":
    main()