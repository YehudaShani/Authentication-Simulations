"""
Symbolic representation of wallet success probabilities.
"""

from helpers.wallet_enumerations import (
    ownerAdvKeysFromStates,
    isCovered,
    SAFE,
    LOST,
    LEAKED,
    STOLEN,
    enumerateStates,
    enumerateStaticWallets,
    walletStrAscii,
)
from collections import Counter
import re
import sympy as sp

# Probability symbols (constants)
P_SAFE, P_LOST, P_LEAKED, P_STOLEN = sp.symbols('S O E T')


def wallet_success_symbolic(wallet, states, ownerStates, advStates):
    """Compute symbolic success probability for a wallet.

    Success occurs when owner can access the wallet AND adversary cannot.

    Args:
        wallet: wallet to evaluate (list of bitmask combinations)
        states: list of state vectors (one per state), where each state vector
                is a list of key states (SAFE/LOST/LEAKED/STOLEN) for each key
        ownerStates: list of owner key combinations for each state (bitmasks)
        advStates: list of adversary key combinations for each state (bitmasks)

    Returns:
        sympy expression: symbolic success probability in terms of
                         P_SAFE, P_LOST, P_LEAKED, P_STOLEN
    """
    symbolic_probs = {
        SAFE: P_SAFE,
        LOST: P_LOST,
        LEAKED: P_LEAKED,
        STOLEN: P_STOLEN,
    }

    # Collect successful state terms
    success_terms = []

    for i, state in enumerate(states):
        owner_ok = isCovered(ownerStates[i], wallet)
        adv_ok = isCovered(advStates[i], wallet)

        if owner_ok and not adv_ok:
            # Count occurrences of each state type
            state_counts = Counter(state)

            # Build symbolic term: product of probabilities
            term = sp.Integer(1)
            for state_type in [SAFE, LOST, LEAKED, STOLEN]:
                count = state_counts.get(state_type, 0)
                if count > 0:
                    term *= symbolic_probs[state_type] ** count

            success_terms.append(term)

    if not success_terms:
        return sp.Integer(0)

    return sum(success_terms)


def simplify_with_constraint(expr):
    """Simplify expression using the sum-to-1 constraint.

    Uses the constraint P_SAFE + P_LOST + P_LEAKED + P_STOLEN = 1
    to simplify the expression by recognizing patterns like:
    P_SAFE * (P_SAFE + P_LOST + P_LEAKED + P_STOLEN) = P_SAFE * 1 = P_SAFE

    Args:
        expr: sympy expression to simplify

    Returns:
        simplified sympy expression with constraint applied
    """
    # The constraint: P_SAFE + P_LOST + P_LEAKED + P_STOLEN = 1
    sum_prob = P_SAFE + P_LOST + P_LEAKED + P_STOLEN

    # First simplify to combine like terms
    result = sp.simplify(expr)

    # Factor the expression to expose common patterns
    # This helps identify factorizations like: P_SAFE * (P_SAFE + P_LOST + ...)
    result = sp.factor(result)

    # Replace occurrences of (P_SAFE + P_LOST + P_LEAKED + P_STOLEN) with 1
    result = result.subs(sum_prob, 1)

    # Simplify again after substitution
    result = sp.simplify(result)

    # Factor again to collect any remaining common terms
    result = sp.factor(result)

    return result


def pprint_expr(expr):
    """Print expression with ^ for powers and proper parentheses.

    Args:
        expr: sympy expression to print
    """
    # Convert to string and replace ** with ^
    expr_str = str(expr)

    # Replace ** with ^
    expr_str = expr_str.replace('**', '^')

    # Add parentheses around powers if needed for clarity
    # Pattern: variable followed by ^number
    # Replace P_VAR^n with (P_VAR)^(n) or P_VAR^(n) for better formatting
    pattern = r'(\w+)\^(\d+)'

    def format_power(match):
        var = match.group(1)
        power = match.group(2)
        # If the variable is part of a larger expression, we might need parentheses
        # For now, just format as var^(power)
        return f'{var}^({power})'

    expr_str = re.sub(pattern, format_power, expr_str)

    print(expr_str)


def group_wallets_by_expression(wallets, keyCount):
    """Group wallets by their identical simplified symbolic expressions.

    Args:
        wallets: list of wallets (each wallet is a list of bitmask combinations)
        keyCount: number of keys in the system

    Returns:
        Dictionary where keys are group indices (integers) and values are dictionaries
        containing:
        - 'expression': the simplified sympy expression for this group
        - 'wallets': list of wallets that have this identical expression
    """
    # Generate all possible states (using dummy probabilities since we just need all states)
    dummy_probs = {SAFE: 0.25, LOST: 0.25, LEAKED: 0.25, STOLEN: 0.25}
    states, _ = enumerateStates(keyCount, dummy_probs)
    ownerStates, advStates = ownerAdvKeysFromStates(states)

    # Dictionary to group wallets by their simplified expression
    # Key: tuple of (expression structure) for reliable comparison
    # Value: dict with 'expression' (sympy expr) and 'wallets' (list)
    groups = {}
    # List to store expressions for comparison
    expression_list = []

    for wallet in wallets:
        # Calculate symbolic expression
        expr = wallet_success_symbolic(wallet, states, ownerStates, advStates)
        
        # Simplify the expression
        simplified_expr = simplify_with_constraint(expr)
        
        # Find matching group by comparing with existing expressions
        # Use sympy's equality check: two expressions are equal if their difference simplifies to 0
        matched = False
        for existing_expr, group_key in expression_list:
            if sp.simplify(simplified_expr - existing_expr) == 0:
                # Found matching expression
                groups[group_key]['wallets'].append(wallet)
                matched = True
                break
        
        if not matched:
            # New unique expression - create new group
            group_key = len(expression_list)
            groups[group_key] = {
                'expression': simplified_expr,
                'wallets': [wallet]
            }
            expression_list.append((simplified_expr, group_key))

    return groups


if __name__ == "__main__":
    print("Example: Wallet symbolic success probability")
    print("=" * 70)

    keyCount = 3
    wallets = enumerateStaticWallets(keyCount)

    # Generate states
    dummy_probs = {SAFE: 0.25, LOST: 0.25, LEAKED: 0.25, STOLEN: 0.25}
    states, _ = enumerateStates(keyCount, dummy_probs)
    ownerStates, advStates = ownerAdvKeysFromStates(states)




    for w in wallets:
        print(f"Wallet: {walletStrAscii(w)}")
        expr = wallet_success_symbolic(w, states, ownerStates, advStates)
        print("Original expression:")
        pprint_expr(expr)
        simplified_expr = simplify_with_constraint(expr)
        print("Simplified expression:")
        pprint_expr(simplified_expr)
        print("-" * 70)
