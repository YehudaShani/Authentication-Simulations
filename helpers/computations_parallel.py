"""
Parallel CPU version of computation functions.
"""
from helpers.wallet_enumerations import SAFE, LOST, LEAKED, STOLEN
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count


def _generate_scenarios_for_a(args):
    """Worker function to generate scenarios for a specific value of 'a' (SAFE).
    
    This function is designed to be picklable for multiprocessing.
    """
    a, n, min_a, max_d, include_zero = args
    
    scenarios = []
    for b in range(n - a + 1):  # LOST
        for c in range(n - a - b + 1):  # LEAKED
            d = n - (a + b + c)  # STOLEN
            
            # Skip if STOLEN exceeds max_theft
            if d > max_d:
                continue
            
            # Convert to exact floats via division by n
            p_safe = a / n
            p_lost = b / n
            p_leaked = c / n
            p_stolen = d / n

            # Enforce positivity constraints if requested
            if not include_zero and (
                p_safe == 0.0 or p_lost == 0.0 or p_leaked == 0.0 or p_stolen == 0.0
            ):
                continue

            # Additional constraint: only keep scenarios where SAFE > STOLEN
            if p_safe <= p_stolen:
                continue

            scenarios.append(
                {
                    SAFE: p_safe,
                    LOST: p_lost,
                    LEAKED: p_leaked,
                    STOLEN: p_stolen,
                }
            )
    
    return scenarios


def generateKeyFaultProbabilityScenariosParallel(
    step=0.05, 
    include_zero=True, 
    min_safe=0.5, 
    max_theft=0.5,
    num_workers=None
):
    """Parallel CPU version of generateKeyFaultProbabilityScenarios.
    
    Generate probability scenarios on an exact integer grid that sum to 1.
    This version parallelizes the outer loop over SAFE values.

    Args:
        step: grid granularity (e.g., 0.5, 0.25, 0.2, 0.1)
        include_zero: if False, excludes scenarios where any probability is 0.0
        min_safe: if provided, enforces SAFE probability >= this threshold
        max_theft: if provided, enforces STOLEN probability <= this threshold
        num_workers: number of worker processes (None = auto-detect CPU count)

    Returns:
        List of probability scenario dictionaries
    """
    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1]")
    if min_safe is not None and (min_safe < 0.0 or min_safe > 1.0):
        raise ValueError("min_safe must be within [0, 1]")
    if max_theft is not None and (max_theft < 0.0 or max_theft > 1.0):
        raise ValueError("max_theft must be within [0, 1]")

    # Use integer grid to avoid floating drift: a+b+c+d = n, probabilities = a/n, ...
    n_float = 1.0 / step
    n = int(round(n_float))
    if abs(n - n_float) > 1e-9:
        # Guard: step must evenly divide 1.0 for an exact grid
        raise ValueError("step must evenly divide 1.0 (e.g., 0.5, 0.25, 0.2, 0.1)")

    # Compute minimal integer count for SAFE given min_safe
    if min_safe is None:
        min_a = 0
    else:
        # Align threshold to grid: a/n >= min_safe  ->  a >= ceil(min_safe * n)
        min_a = int(math.ceil(min_safe * n - 1e-12))
        if min_a > n:
            return []
    
    # Compute maximal integer count for STOLEN given max_theft
    if max_theft is None:
        max_d = n
    else:
        # Align threshold to grid: d/n <= max_theft  ->  d <= floor(max_theft * n)
        max_d = int(math.floor(max_theft * n + 1e-12))
        if max_d < 0:
            return []

    if num_workers is None:
        num_workers = cpu_count()
    
    # Prepare arguments for each value of 'a'
    a_values = list(range(min_a, n + 1))
    total_a_values = len(a_values)
    
    print(f"Generating scenarios with {num_workers} CPU cores...")
    print(f"  Processing {total_a_values} values of 'a' (SAFE) in parallel")
    
    all_scenarios = []
    completed = 0
    
    # Process each value of 'a' in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_a = {
            executor.submit(_generate_scenarios_for_a, (a, n, min_a, max_d, include_zero)): a
            for a in a_values
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_a):
            a_value = future_to_a[future]
            try:
                scenarios_for_a = future.result()
                all_scenarios.extend(scenarios_for_a)
                
                completed += 1
                if completed % max(1, total_a_values // 10) == 0 or completed == total_a_values:
                    print(f"  Completed {completed}/{total_a_values} values of 'a' "
                          f"({len(all_scenarios)} scenarios generated so far)")
                    
            except Exception as exc:
                print(f"Value a={a_value} generated an exception: {exc}")
                raise
    
    print(f"Generated {len(all_scenarios)} total scenarios")
    return all_scenarios

