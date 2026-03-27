import copy
import itertools

from consts import SAFE, STOLEN, LOST, LEAKED, KeyStates, KeyStateString


# Check if the key combination can access any wallet combination
def isCovered(keyCombination, wallet):
    """Is any of the wallet combinations covered by the keyCombination?
    (Then keyCombination is redundant)
    """
    for walletCombination in wallet:
        if walletCombination & keyCombination == walletCombination:
            return True

    return False


# Build all possible static wallets for keyCount keys
def enumerateStaticWallets(keyCount, deduplicate_by_architecture=True):
    # All possible key combinations (from 001, 010, 011, ..., 111)
    wallets = enumerateStaticSubWallets(baseWallet=[], prevCombi=0, keyCount=keyCount)
    if deduplicate_by_architecture:
        wallets = deduplicateWalletsByArchitecture(wallets, keyCount)
    return wallets


# @Memoized()
def enumerateStaticSubWallets(baseWallet, prevCombi, keyCount):
    wallets = []
    for currCombi in range(prevCombi + 1, 2 ** keyCount):
        if not isCovered(currCombi, baseWallet):
            currWallet = baseWallet + [currCombi]
            # # Skip this combination:
            # wallets += enumerateStaticSubWallets(baseWallet, currCombi, keyCount)
            # Just this combination:
            wallets += [copy.copy(currWallet)]
            # This combination and all its subwallets:
            wallets += enumerateStaticSubWallets(currWallet, currCombi, keyCount)

    return wallets


# @Memoized()
# Enumerate all possible states for keyCount keys, stating the probability of each state
def enumerateStates(keyCount, keyStateProbabilities):
    """Create a list of lists, each list contains a series of the states of all the keyCount keys.
    keyStateProbabilities is a dictionary from Key state (SAFE, LOST,...) to probability.
    """
    if keyCount <= 0:
        raise Exception("Invalid number of keys")
    if keyCount == 1:
        states = [[state] for state in sorted(KeyStates)]
        probabilities = [keyStateProbabilities[state] for state in sorted(KeyStates)]
        return states, probabilities

    assert set(keyStateProbabilities.keys()) == set(KeyStates)
    states = []
    probabilities = []
    stateSuffix, suffixesProbabilities = enumerateStates(
        keyCount - 1, keyStateProbabilities
    )
    for keyState in KeyStates:
        for iSuffix in range(len(stateSuffix)):
            suffix = stateSuffix[iSuffix]
            pSuffix = suffixesProbabilities[iSuffix]
            states.append([keyState] + suffix)
            if keyStateProbabilities:
                probabilities.append(keyStateProbabilities[keyState] * pSuffix)

    return states, probabilities


def ownerAdvKeysFromStates(states):
    ownerStates = []
    advStates = []

    for state in states:
        ownerState = 0
        advState = 0
        i = 0  # index of digit
        for keyState in state:
            bools = {
                SAFE: (1, 0),  # (True, False),
                LOST: (0, 0),  # (False, False),
                LEAKED: (1, 1),  # (True, True),
                STOLEN: (0, 1),  # (False, True),
            }[keyState]
            ownerState += bools[0] * 2 ** i
            advState += bools[1] * 2 ** i
            i += 1
        ownerStates.append(ownerState)
        advStates.append(advState)

    return ownerStates, advStates


def walletStr(wallet):
    combiStrings = []
    for combi in wallet:
        keyIndices = oneBitIndices(combi)
        combiStrings.append("(" + " ∧ ".join(keyIndices) + ")")
    walletString = " ∨ ".join(combiStrings)
    return walletString


def walletStrAscii(wallet, and_token="AND", or_token="OR"):
    """ASCII-only wallet representation for CSV/Excel compatibility.

    Example: (1 AND 2) OR (3)
    """
    combiStrings = []
    for combi in wallet:
        keyIndices = oneBitIndices(combi)
        combiStrings.append("(" + f" {and_token} ".join(keyIndices) + ")")
    walletString = f" {or_token} ".join(combiStrings)
    return walletString


# https://stackoverflow.com/a/49592515/385482
def oneBitIndices(number):
    bits = []
    for i, c in enumerate(bin(number)[:1:-1], 1):
        if c == "1":
            bits.append(str(i))
    return bits


def permuteBits(number, permutation):
    """Apply a permutation of bit positions to an integer bitmask.

    permutation is a tuple/list where index i maps bit (i) to bit (permutation[i]).
    Bit positions are 1-based in our logic (LSB is position 1).
    """
    result = 0
    for i, target_pos in enumerate(permutation, start=1):
        if number & (1 << (i - 1)):
            result |= (1 << (target_pos - 1))
    return result


def canonicalizeWallet(wallet, keyCount):
    """Return a canonical representation of a wallet up to key index renaming.

    We test all permutations of key indices (1..keyCount) and choose the
    lexicographically smallest sorted tuple of permuted combinations.
    """
    indices = tuple(range(1, keyCount + 1))
    best = None
    for perm in itertools.permutations(indices):
        transformed = tuple(sorted(permuteBits(c, perm) for c in wallet))
        if best is None or transformed < best:
            best = transformed
    return best


def deduplicateWalletsByArchitecture(wallets, keyCount):
    """Deduplicate wallets that are identical up to key renaming.

    Useful when all keys are homogeneous and success probability depends only
    on structure, not on which specific keys are used.
    """
    seen = set()
    unique = []
    for wallet in wallets:
        canon = canonicalizeWallet(wallet, keyCount)
        if canon not in seen:
            seen.add(canon)
            unique.append(wallet)
    return unique

def stateStr(state):
    """Convert a state list to a readable string format.
    
    Args:
        state: list of key states (SAFE/LOST/LEAKED/STOLEN)
    
    Returns:
        String representation like "[Safe, Loss, Leak, Theft]"
    """
    state_names = {
        SAFE: "Safe",
        LOST: "Loss",
        LEAKED: "Leak",
        STOLEN: "Theft"
    }
    state_strings = [state_names[key_state] for key_state in state]
    return "[" + ", ".join(state_strings) + "]"

def walletDifferingScenarios(wallet1, wallet2, keyCount, keyStateProbabilities1=None, keyStateProbabilities2=None):
    """Return the scenarios (states) in which two wallets have different outcomes.

    A wallet succeeds in a scenario when: owner can access AND adversary cannot access.
    Two wallets differ in a scenario if one succeeds and the other doesn't.

    Args:
        wallet1: first wallet (list of bitmask combinations)
        wallet2: second wallet (list of bitmask combinations)
        keyCount: number of keys in the system
        keyStateProbabilities1: probability dictionary for evaluating wallet1 (default: equal probabilities)
        keyStateProbabilities2: probability dictionary for evaluating wallet2 (default: same as keyStateProbabilities1)

    Returns:
        Tuple of two lists:
        - First list: scenarios where wallet1 succeeds but wallet2 fails
        - Second list: scenarios where wallet2 succeeds but wallet1 fails
        Each state vector is a list of key states (SAFE/LOST/LEAKED/STOLEN) for each key.
    """
    # Use default probabilities if not provided
    if keyStateProbabilities1 is None:
        keyStateProbabilities1 = {SAFE: 0.25, LOST: 0.25, LEAKED: 0.25, STOLEN: 0.25}
    if keyStateProbabilities2 is None:
        keyStateProbabilities2 = keyStateProbabilities1
    
    # Generate all possible states and their probabilities for each wallet
    # Note: states will be the same regardless of probabilities, only probabilities differ
    states1, probabilities1 = enumerateStates(keyCount, keyStateProbabilities1)
    states2, probabilities2 = enumerateStates(keyCount, keyStateProbabilities2)
    ownerStates1, advStates1 = ownerAdvKeysFromStates(states1)
    ownerStates2, advStates2 = ownerAdvKeysFromStates(states2)

    wallet1_succeeds = []
    wallet2_succeeds = []
    wallet1_total_prob = 0.0
    wallet2_total_prob = 0.0
    
    # Print header
    print("=" * 80)
    print("Scenarios where wallets differ:")
    print("=" * 80)
    print(f"\nScenarios where wallet1 succeeds but wallet2 fails:")
    print("-" * 80)
    
    # Iterate once through all states
    for i, state in enumerate(states1):
        # Check if wallet1 succeeds in this scenario
        owner_ok1 = isCovered(ownerStates1[i], wallet1)
        adv_ok1 = isCovered(advStates1[i], wallet1)
        success1 = owner_ok1 and not adv_ok1
        if success1:
            wallet1_total_prob += probabilities1[i]

        # Check if wallet2 succeeds in this scenario (using same state index)
        owner_ok2 = isCovered(ownerStates2[i], wallet2)
        adv_ok2 = isCovered(advStates2[i], wallet2)
        success2 = owner_ok2 and not adv_ok2
        if success2:
            wallet2_total_prob += probabilities2[i]

        # Separate scenarios where one succeeds and the other fails
        if success1 and not success2:
            wallet1_succeeds.append((state, probabilities1[i]))
        elif success2 and not success1:
            wallet2_succeeds.append((state, probabilities2[i]))
    
    # Sort by probability (descending - highest probability first)
    wallet1_succeeds.sort(key=lambda x: x[1], reverse=True)
    wallet2_succeeds.sort(key=lambda x: x[1], reverse=True)
    
    # Print wallet1 scenarios
    for state, prob in wallet1_succeeds:
        print(f"  State: {stateStr(state)}, Probability: {prob:.6f}")
    
    print(f"\nScenarios where wallet2 succeeds but wallet1 fails:")
    print("-" * 80)
    for state, prob in wallet2_succeeds:
        print(f"  State: {stateStr(state)}, Probability: {prob:.6f}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary:")
    print("-" * 80)
    print(f"Wallet1 total success probability: {wallet1_total_prob:.6f}")
    print(f"Wallet2 total success probability: {wallet2_total_prob:.6f}")
    print(f"Difference (wallet1 - wallet2): {wallet1_total_prob - wallet2_total_prob:.6f}")
    print("=" * 80)
    
    # Return just the states (without probabilities) for backward compatibility
    return [state for state, _ in wallet1_succeeds], [state for state, _ in wallet2_succeeds]


if __name__ == "__main__":
    # Test walletDifferingScenarios
    keyCount = 4
    wallet_1 = [5,3,6]
    wallet_2 = [5,3,9,14]
    
    # Define probability distributions (can be different for each wallet)
    probs1 = {SAFE: 0.5, LOST: 0.3, LEAKED: 0.1, STOLEN: 0.1}
    probs2 = probs1
    
    wallet1_succeeds, wallet2_succeeds = walletDifferingScenarios(
        wallet_1, wallet_2, keyCount, 
        keyStateProbabilities1=probs1, 
        keyStateProbabilities2=probs2
    )