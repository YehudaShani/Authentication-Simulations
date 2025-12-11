import csv
from helpers.computations import findOptimalWallet
from helpers.wallet_enumerations import enumerateStaticWallets, SAFE, LOST, LEAKED, STOLEN, walletStrAscii

cases = [
    {
        SAFE: 0.5,
        LOST: 0.3,
        LEAKED: 0.1,
        STOLEN: 0.1
    },
    {
        SAFE: 0.4,
        LOST: 0.2,
        LEAKED: 0.2,
        STOLEN: 0.2
    },
    {
        SAFE: 0.6,
        LOST: 0.2,
        LEAKED: 0.15,
        STOLEN: 0.05
    },
    {
        SAFE: 0.3,
        LOST: 0.3,
        LEAKED: 0.25,
        STOLEN: 0.15
    },
    {
        SAFE: 0.7,
        LOST: 0.15,
        LEAKED: 0.1,
        STOLEN: 0.05
    },
    {
        SAFE: 0.35,
        LOST: 0.25,
        LEAKED: 0.2,
        STOLEN: 0.2
    },
    {
        SAFE: 0.55,
        LOST: 0.2,
        LEAKED: 0.15,
        STOLEN: 0.1
    },
    {
        SAFE: 0.45,
        LOST: 0.3,
        LEAKED: 0.15,
        STOLEN: 0.1
    },
    {
        SAFE: 0.65,
        LOST: 0.15,
        LEAKED: 0.12,
        STOLEN: 0.08
    },
    {
        SAFE: 0.5,
        LOST: 0.25,
        LEAKED: 0.15,
        STOLEN: 0.1
    }
]


key_counts = [2, 3, 4, 5]
# Use a dictionary to map key_count to wallets
wallets = {key_count: enumerateStaticWallets(key_count) for key_count in key_counts}

# Collect results for Excel export
results = []

for case_idx, case in enumerate(cases):
    # print optimal wallets for each key count
    for key_count_idx, key_count in enumerate(key_counts):
        print(f"Checking optimal wallets for probability: {case}")
        optimal_wallets, best_prob = findOptimalWallet(wallets[key_count], key_count, case)
        optimal_wallets_str = [walletStrAscii(w) for w in optimal_wallets]
        print(f"Optimal wallets for key count {key_count}: {optimal_wallets_str} with probability {best_prob}")
        
        # Store results - only show case info on first row of each case
        is_first_row = (key_count_idx == 0)
        results.append({
            'Case': case_idx + 1 if is_first_row else '',
            'SAFE': case[SAFE] if is_first_row else '',
            'LOST': case[LOST] if is_first_row else '',
            'LEAKED': case[LEAKED] if is_first_row else '',
            'STOLEN': case[STOLEN] if is_first_row else '',
            'Key_Count': key_count,
            'Optimal_Wallets': ' | '.join(optimal_wallets_str) if len(optimal_wallets_str) > 1 else optimal_wallets_str[0],
            'Best_Probability': best_prob
        })
    
    # Add empty row between cases (but not after the last case)
    if case_idx < len(cases) - 1:
        results.append({
            'Case': '',
            'SAFE': '',
            'LOST': '',
            'LEAKED': '',
            'STOLEN': '',
            'Key_Count': '',
            'Optimal_Wallets': '',
            'Best_Probability': ''
        })

# Save to Excel using openpyxl
try:
    from openpyxl import Workbook
    
    wb = Workbook()
    ws = wb.active
    
    # Define column headers
    headers = ['Case', 'SAFE', 'LOST', 'LEAKED', 'STOLEN', 'Key_Count', 'Optimal_Wallets', 'Best_Probability']
    ws.append(headers)
    
    # Write data rows
    for result in results:
        row = [
            result['Case'],
            result['SAFE'],
            result['LOST'],
            result['LEAKED'],
            result['STOLEN'],
            result['Key_Count'],
            result['Optimal_Wallets'],
            result['Best_Probability']
        ]
        ws.append(row)
    
    output_file = 'optimal_wallets_results.xlsx'
    wb.save(output_file)
    print(f"\nResults saved to {output_file}")
    print(f"Total rows: {len(results)}")
    
except ImportError:
    # Fallback: save as CSV
    output_file = 'optimal_wallets_results_up_to_6_keys.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Case', 'SAFE', 'LOST', 'LEAKED', 'STOLEN', 'Key_Count', 'Optimal_Wallets', 'Best_Probability']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output_file} (CSV format)")
    print(f"Total rows: {len(results)}")
    print("Note: Install openpyxl (pip install openpyxl) to save as Excel format")
except Exception as e:
    print(f"\nError saving to Excel: {e}")
    import traceback
    traceback.print_exc()
    # Fallback to CSV
    output_file = 'optimal_wallets_results_up_to_6_keys.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Case', 'SAFE', 'LOST', 'LEAKED', 'STOLEN', 'Key_Count', 'Optimal_Wallets', 'Best_Probability']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {output_file} (CSV format) as fallback")