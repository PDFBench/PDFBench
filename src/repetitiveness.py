import json
import multiprocessing as mp
import os
from typing import Optional

import numpy as np
from tqdm.auto import tqdm
from transformers import (
    logging,
)

logging.set_verbosity_error()


def compute_repetitiveness(sequence: str) -> float:
    """
    Calculate the proportion of characters in a string that are repeated substrings
    :param sequence: protein sequence
    :return: the propertion of repeated characters, range [0.0, 1.0]
    """
    n = len(sequence)

    # Bound check
    if n == 0:
        return 0.0

    # Process
    regions = []
    max_window_size = min(20, n // 2)  # limit the window size
    for window_size in range(1, max_window_size + 1):
        for i in range(n - window_size + 1):
            pattern = sequence[i : i + window_size]
            count = 1

            j = i + window_size
            while j <= n - window_size:
                next_segment = sequence[j : j + window_size]
                if next_segment == pattern:
                    count += 1
                    j += window_size
                else:
                    break

            # Only keep regions with at least 3 repetitions
            if count >= 3:
                start = i
                end = i + window_size * count
                regions.append((start, end))

    # Bound check
    if not regions:
        return 0.0

    # Sort and merge
    sorted_regions = sorted(regions, key=lambda x: x[0])
    merged = []
    for start, end in sorted_regions:
        if not merged:
            merged.append([start, end])
        else:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1][1] = max(last_end, end)
            else:
                merged.append([start, end])

    # Return
    total_repeat = sum(end - start for start, end in merged)
    proportion = total_repeat / n
    return proportion


def compute_repeatN(sequence: str, n: int):
    if len(sequence) == n:
        return 0.0

    indeps = set()
    for i in range(len(sequence) - n + 1):
        indeps.add(sequence[i : i + n])

    return max(0.0, 1 - len(indeps) / (len(sequence) - n))


def _main(uid: int, queue: mp.Queue, subset: list):
    results: list = [dict() for _ in range(len(subset))]

    # region non-model based
    idx = 0
    for item in tqdm(
        subset,
        desc=f"Process {uid} - Repetitiveness",
        position=uid + 1,
        ncols=100,
    ):
        mutant = item["response"]

        res: dict = {
            "sequence": mutant,
            "repeat_2": compute_repeatN(mutant, 2),
            # "repeat_3": compute_repeatN(mutant, 3),
            "repeat_5": compute_repeatN(mutant, 5),
            # "repeat_10": compute_repeatN(mutant, 10),
            "repetitiveness": compute_repetitiveness(mutant),
        }
        results[idx].update(res)
        idx += 1

    queue.put(results)


def main(
    num_workers: int = 4,
    sequence_file: Optional[str] = None,
    evaluation_file: Optional[str] = None,
):
    assert sequence_file and evaluation_file

    if not os.path.exists(evaluation_file):
        mp.set_start_method("spawn", force=True)

        with open(sequence_file, "r") as f:
            data: list = json.load(f)

        queue: mp.Queue = mp.Queue()
        processes: list = []
        for i in range(num_workers):
            piece = len(data) // num_workers
            beg_idx = i * piece
            end_idx = (i + 1) * piece if i != num_workers - 1 else len(data)
            subset = data[beg_idx:end_idx]

            p = mp.Process(target=_main, args=(i, queue, subset))
            p.start()
            processes.append(p)

        results: list = [queue.get() for _ in range(len(processes))]
        results = [element for sublist in results for element in sublist]

        for p in processes:
            p.join()

        with open(evaluation_file, "w") as f:
            json.dump(results, f, indent=4)  # type: ignore
    else:
        print("Load processed evaluation file")
        with open(evaluation_file, "r") as f:
            results: list = json.load(f)

    support_metrics = [
        "repeat_2",
        # "repeat_3",
        "repeat_5",
        # "repeat_10",
        "repetitiveness",
    ]
    for metric in support_metrics:
        mean = np.mean([sample[metric] for sample in results])
        print(f"mean {metric}: {mean:.2f}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
