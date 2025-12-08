from experiments.experiment_optimal_wallets import count_optimal_wallet_occurrences
from helpers.computations import generateKeyFaultProbabilityScenarios
from helpers.wallet_enumerations import enumerateStaticWallets


def main():
    # Configuration
    keyCount = 4
    num_probabilities = 100000
    step = 0.01  # Small step to generate many scenarios
    min_safe = 0.1
    deduplicate_by_architecture = True
    
    print(f"Generating {num_probabilities} probability scenarios...")
    print(f"  step={step}, min_safe={min_safe}, keyCount={keyCount}")
    
    # Generate probability scenarios
    all_scenarios = generateKeyFaultProbabilityScenarios(
        step=step,
        include_zero=False,
        min_safe=min_safe
    )
    
    # Take the first num_probabilities scenarios
    probabilities_list = all_scenarios[:num_probabilities]
    print(f"Using {len(probabilities_list)} probability scenarios")
    
    # Generate wallets
    print(f"Generating wallets for keyCount={keyCount}...")
    wallets = enumerateStaticWallets(
        keyCount, 
        deduplicate_by_architecture=deduplicate_by_architecture
    )
    print(f"Generated {len(wallets)} wallets")
    
    # Count optimal wallet occurrences
    print("\nCounting optimal wallet occurrences...")
    results = count_optimal_wallet_occurrences(
        probabilities_list=probabilities_list,
        wallets=wallets,
        keyCount=keyCount,
        output_csv_path="optimal_wallet_occurrences_100000.csv",
        print_results=True
    )
    
    print(f"\nAnalysis complete! Results saved to optimal_wallet_occurrences_100000.csv")


if __name__ == "__main__":
    main()