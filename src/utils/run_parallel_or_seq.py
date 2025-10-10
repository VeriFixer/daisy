import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def run_parallel_or_seq(items, task_fn, desc, *task_args, parallel=True):
    """
    Runs `task_fn(item, *task_args)` either in parallel or sequentially.
    
    Args:
        items (iterable): List of work items.
        task_fn (callable): Function that takes (item, *task_args).
        desc (str): Description for tqdm progress bar.
        *task_args: Extra arguments to pass to the task function.
        parallel (bool): Whether to use threads or run sequentially.
    
    Returns:
        List of results (or None for failed tasks).
    """
    results = []
    PHYSICAL_CORES = os.cpu_count() // 2
    SAFE_THREADS = max(1, PHYSICAL_CORES - 1)

    if parallel:
        with ThreadPoolExecutor(max_workers=SAFE_THREADS) as executor:
            futures = {
                executor.submit(task_fn, item, *task_args): item
                for item in items
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                item = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"[Warning] Error processing {item}: {e}")
    else:
        for item in tqdm(items, desc=f"{desc} (sequential)"):
            try:
                results.append(task_fn(item, *task_args))
            except Exception as e:
                print(f"[Warning] Error processing {item}: {e}")

    return results