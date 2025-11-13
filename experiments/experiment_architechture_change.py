from collections import defaultdict
import pandas as pd

from helpers.computations import findOptimalWallet, generateKeyFaultProbabilityScenarios
from helpers.wallet_enumerations import walletStrAscii, enumerateStaticWallets
from consts import SAFE, LOST, LEAKED, STOLEN




def map_wallets_key_count_change(
    probabilities,
    keyCount1,
    keyCount2,
    deduplicate_by_architecture=True,
):
    """Map optimal wallets from keyCount1 to optimal wallets in keyCount2.
    
    Returns a dictionary where each key is an optimal wallet from keyCount1 (as string),
    and each value is a list of optimal wallets from keyCount2 (as strings).
    This shows which optimal wallets can be achieved in keyCount2 when starting
    from each optimal wallet in keyCount1.
    
    Args:
        probabilities: dict with SAFE/LOST/LEAKED/STOLEN probabilities
        keyCount1: smaller key count (e.g., 2)
        keyCount2: larger key count (e.g., 4)
        deduplicate_by_architecture: whether to deduplicate wallets by architecture
    
    Returns:
        dict mapping wallet strings from keyCount1 to list of wallet strings from keyCount2
        Format: {wallet_str_keyCount1: [wallet_str_keyCount2, ...], ...}
    """
    # Generate wallet groups for both key counts
    wallet_group1 = enumerateStaticWallets(keyCount1, deduplicate_by_architecture=deduplicate_by_architecture)
    wallet_group2 = enumerateStaticWallets(keyCount2, deduplicate_by_architecture=deduplicate_by_architecture)

    optimal_wallets_1, _ = findOptimalWallet(wallet_group1, keyCount1, probabilities)
    optimal_wallets_1_str = [walletStrAscii(w) for w in optimal_wallets_1]
    # Find optimal wallets for keyCount2
    optimal_wallets_2, _ = findOptimalWallet(wallet_group2, keyCount2, probabilities)
    
    # Convert optimal wallets from keyCount2 to strings
    optimal_wallets_2_str = [walletStrAscii(w) for w in optimal_wallets_2]
    
    # Create mapping: each optimal wallet from keyCount1 maps to all optimal wallets from keyCount2
    mapping = {}
    for wallet1 in optimal_wallets_1:
        wallet1_str = walletStrAscii(wallet1)
        mapping[wallet1_str] = optimal_wallets_2_str.copy()
    
    return mapping


def count_wallet_transitions_across_scenarios(
    keyCount1,
    keyCount2,
    probabilities_list,
    deduplicate_by_architecture=True,
    output_excel_path=None,
    count_ties_separately=False,
):
    """Count how many probability scenarios lead from each wallet in group1 to each wallet in group2.
    
    For each probability scenario, finds optimal wallets in both groups and counts
    how many times each wallet1 -> wallet2 transition occurs.
    
    Args:
        keyCount1: smaller key count (e.g., 2)
        keyCount2: larger key count (e.g., 4)
        probabilities_list: list of probability dicts (each with SAFE/LOST/LEAKED/STOLEN)
        deduplicate_by_architecture: whether to deduplicate wallets by architecture
        output_excel_path: path to save Excel file (if None, returns DataFrame without saving)
        count_ties_separately: if True, track unique and tied scenarios separately
    
    Returns:
        If count_ties_separately=False:
            pandas DataFrame with rows = wallets from keyCount2, columns = wallets from keyCount1,
            values = count of scenarios where this transition occurred
        If count_ties_separately=True:
            dict with keys 'unique', 'tied', 'total', each containing a DataFrame
    """
    # Generate wallet groups for both key counts
    wallet_group1 = enumerateStaticWallets(keyCount1, deduplicate_by_architecture=deduplicate_by_architecture)
    wallet_group2 = enumerateStaticWallets(keyCount2, deduplicate_by_architecture=deduplicate_by_architecture)
    
    # Get all possible wallets as strings for consistent indexing
    all_wallets_1 = [walletStrAscii(w) for w in wallet_group1]
    all_wallets_2 = [walletStrAscii(w) for w in wallet_group2]
    
    # Initialize counting structure: wallet1 -> wallet2 -> count
    transition_counts = defaultdict(lambda: defaultdict(int))
    unique_transition_counts = defaultdict(lambda: defaultdict(int))
    tied_transition_counts = defaultdict(lambda: defaultdict(int))
    
    # Process each probability scenario
    total_scenarios = len(probabilities_list)
    print(f"Processing {total_scenarios} probability scenarios...")
    
    for idx, probabilities in enumerate(probabilities_list):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{total_scenarios} scenarios...")
        
        # Find optimal wallets for both groups
        optimal_wallets_1, _ = findOptimalWallet(wallet_group1, keyCount1, probabilities)
        optimal_wallets_2, _ = findOptimalWallet(wallet_group2, keyCount2, probabilities)
        
        # Convert to strings
        optimal_wallets_1_str = [walletStrAscii(w) for w in optimal_wallets_1]
        optimal_wallets_2_str = [walletStrAscii(w) for w in optimal_wallets_2]
        
        # Determine if this is a tie scenario
        is_tie = len(optimal_wallets_1_str) > 1 or len(optimal_wallets_2_str) > 1
        
        # Count transitions: each optimal wallet in group1 maps to all optimal wallets in group2
        for wallet1_str in optimal_wallets_1_str:
            for wallet2_str in optimal_wallets_2_str:
                transition_counts[wallet1_str][wallet2_str] += 1
                if count_ties_separately:
                    if is_tie:
                        tied_transition_counts[wallet1_str][wallet2_str] += 1
                    else:
                        unique_transition_counts[wallet1_str][wallet2_str] += 1
    
    print(f"Completed processing {total_scenarios} scenarios.")
    
    def create_dataframe(counts_dict):
        """Helper function to create DataFrame from counts dictionary."""
        data = []
        for wallet1_str in all_wallets_1:
            row = {'Wallet_Group1': wallet1_str}
            for wallet2_str in all_wallets_2:
                count = counts_dict[wallet1_str][wallet2_str]
                row[wallet2_str] = count
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.set_index('Wallet_Group1')
        # Transpose: rows = wallets from group2, columns = wallets from group1
        df = df.T
        return df
    
    # Create main DataFrame
    df = create_dataframe(transition_counts)
    
    if count_ties_separately:
        # Create separate DataFrames for unique and tied scenarios
        df_unique = create_dataframe(unique_transition_counts)
        df_tied = create_dataframe(tied_transition_counts)
        
        # Save to Excel if path provided
        if output_excel_path:
            with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Total')
                df_unique.to_excel(writer, sheet_name='Unique')
                df_tied.to_excel(writer, sheet_name='Tied')
            print(f"\nSaved results to {output_excel_path}")
            print(f"  Total DataFrame shape: {df.shape}")
            print(f"  Unique DataFrame shape: {df_unique.shape}")
            print(f"  Tied DataFrame shape: {df_tied.shape}")
        
        return {
            'total': df,
            'unique': df_unique,
            'tied': df_tied,
        }
    else:
        # Save to Excel if path provided
        if output_excel_path:
            df.to_excel(output_excel_path, sheet_name='Wallet_Transitions')
            print(f"\nSaved results to {output_excel_path}")
            print(f"DataFrame shape: {df.shape} (rows = wallets in keyCount {keyCount2}, columns = wallets in keyCount {keyCount1})")
        
        return df


def main():
    """
    Example main function to demonstrate architecture change experiment.
    Modify the input parameters as required for your use case.
    """
    # Example: Changing from 2-key group to 4-key group (or other types/architectures)
    keyCount1 = 2
    keyCount2 = 4

    min_safe = 0.2  # Minimum SAFE probability for scenarios
    step = 0.02
    # Generate probability scenarios
    print("Generating probability scenarios...")
    probabilities_list = generateKeyFaultProbabilityScenarios(
        step=step,
        include_zero=False,
        min_safe=min_safe
    )
    print(f"Generated {len(probabilities_list)} probability scenarios.")
    
    # Count wallet transitions across all scenarios and save to Excel
    print("\n" + "=" * 80)
    print(f"Counting wallet transitions from keyCount {keyCount1} to keyCount {keyCount2}")
    print("=" * 80)
    
    result = count_wallet_transitions_across_scenarios(
        keyCount1=keyCount1,
        keyCount2=keyCount2,
        probabilities_list=probabilities_list,
        output_excel_path=f"wallet_transitions_{keyCount1}to{keyCount2}_minsafe={min_safe}_step={step}.xlsx",
        count_ties_separately=True,
    )
    
    # Handle return value (either DataFrame or dict)
    if isinstance(result, dict):
        df = result['total']
        df_unique = result['unique']
        df_tied = result['tied']
        
        print(f"\nSummary (Total):")
        print(f"  Total wallets in keyCount {keyCount1}: {len(df.columns)}")
        print(f"  Total wallets in keyCount {keyCount2}: {len(df.index)}")
        print(f"  Total transitions counted: {df.sum().sum()}")
        
        print(f"\nSummary (Unique - no ties):")
        print(f"  Unique transitions counted: {df_unique.sum().sum()}")
        
        print(f"\nSummary (Tied - at least one group has multiple optimal wallets):")
        print(f"  Tied transitions counted: {df_tied.sum().sum()}")
        
        print(f"\nTop 5 most common transitions (Total):")
        transitions = []
        for wallet2 in df.index:
            for wallet1 in df.columns:
                count = df.loc[wallet2, wallet1]
                if count > 0:
                    transitions.append((wallet1, wallet2, count))
        transitions.sort(key=lambda x: x[2], reverse=True)
        for i, (w1, w2, count) in enumerate(transitions[:5], 1):
            print(f"  {i}. {w1} → {w2}: {count} scenarios")
    else:
        df = result
        print(f"\nSummary:")
        print(f"  Total wallets in keyCount {keyCount1}: {len(df.columns)}")
        print(f"  Total wallets in keyCount {keyCount2}: {len(df.index)}")
        print(f"  Total transitions counted: {df.sum().sum()}")
        print(f"\nTop 5 most common transitions:")
        # Flatten and sort by count (note: df is transposed, so wallet2 is index, wallet1 is column)
        transitions = []
        for wallet2 in df.index:
            for wallet1 in df.columns:
                count = df.loc[wallet2, wallet1]
                if count > 0:
                    transitions.append((wallet1, wallet2, count))
        transitions.sort(key=lambda x: x[2], reverse=True)
        for i, (w1, w2, count) in enumerate(transitions[:5], 1):
            print(f"  {i}. {w1} → {w2}: {count} scenarios")

if __name__ == "__main__":
    main()
