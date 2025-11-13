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
def enumerateStaticWallets(keyCount, deduplicate_by_architecture=False):
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



if __name__ == "__main__":
    # Test enumerateStaticWallets
    keyCount = 3
    wallets = enumerateStaticWallets(keyCount, deduplicate_by_architecture=True)
    print(f"There are {len(wallets)} combinations of wallet for {keyCount} keys:")
    for wallet in wallets:
        print(walletStr(wallet))