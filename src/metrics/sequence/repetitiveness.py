import multiprocessing as mp

import numpy as np
from tqdm.auto import tqdm

from src.configs.sequence_args import Repeat_Algorithm
from src.metrics import BaseEvaluator, BaseMetric
from src.utils.multiprocess import multiprocess_evaluate

def compute_repeat(sequence: str) -> float:
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
    if len(sequence) <= n:
        return 0.0

    indeps = set()
    for i in range(len(sequence) - n + 1):
        indeps.add(sequence[i : i + n])

    return max(0.0, 1 - len(indeps) / (len(sequence) - n + 1))


def repeat_evaluate_worker(
    queue: mp.Queue, pid: int, subset: list, **kwargs
) -> None:
    design_batch_size = kwargs.get("design_batch_size")
    compute_methods = kwargs.get("compute_methods")
    repn = kwargs.get("RepN")
    if not (
        design_batch_size is not None
        and compute_methods is not None
        and repn is not None
    ):
        raise ValueError(
            "Invalid kwargs: \n"
            f"design_batch_size: {design_batch_size}\n"
            f"compute_methods: {compute_methods}\n"
            f"repn: {repn}"
        )
    items = tqdm(
        subset,
        desc="Repetitiveness",
        position=pid + 1,
        ncols=100,
        disable=pid != 0,
    )
    results: list = [dict() for _ in range(len(subset))]
    for idx, item in enumerate(items):
        res = {
            "instruction": item["instruction"],
            "reference": item["reference"],
        }
        for b in range(1, design_batch_size + 1):
            response = item[f"response#{b}"]

            res.update({f"response#{b}": response})
            if Repeat_Algorithm.Repeat.name in compute_methods:
                res.update({f"repeat#{b}": compute_repeat(sequence=response)})
            if Repeat_Algorithm.RepN.name in compute_methods:
                res.update(
                    {
                        f"rep{n}#{b}": compute_repeatN(sequence=response, n=n)
                        for n in repn
                    }
                )
        results[idx].update(res)

    queue.put((pid, results))


class RepetitivenessMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
        self.RepN = config.repeat.RepN
        self.compute_methods = config.repeat.compute_methods
        self._name = config.repeat.name

    @property
    def metrics(self) -> list[str]:
        _metrics = []
        if Repeat_Algorithm.RepN.name in self.compute_methods:
            _metrics.extend([f"rep_{n}" for n in self.RepN])
        if Repeat_Algorithm.Repeat.name in self.compute_methods:
            _metrics.append("repeat")
        return _metrics

    def summary(self, results) -> dict:
        bs = self.design_batch_size
        if bs == 1:
            return {
                "repeat": results["repeat#1"].mean() * 100,
                **{
                    f"rep{n}": results[f"rep{n}#1"].mean() * 100
                    for n in self.RepN
                },
            }
        else:
            repeats = [
                results[f"repeat#{b}"].mean() * 100 for b in range(1, bs + 1)
            ]
            repns = [
                [results[f"rep{n}#{b}"].mean() * 100 for b in range(1, bs + 1)]
                for n in self.RepN
            ]

            out = {
                "repeat": (
                    rf"{np.mean(repeats):.2f}"
                    r"\(\pm\)"
                    rf"{np.std(repeats, ddof=1):.2f}"
                ),
                **{f"repeat#{b}": repeats[b - 1] for b in range(1, bs + 1)},
            }

            for idx, n in enumerate(self.RepN):
                out[f"rep{n}"] = (
                    rf"{np.mean(repns[idx]):.2f}"
                    r"\(\pm\)"
                    rf"{np.std(repns[idx], ddof=1):.2f}"
                )
                out.update(
                    {f"rep{n}#{b}": repns[idx][b - 1] for b in range(1, bs + 1)}
                )

            return out


class RepetitivenessEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.RepN = config.repeat.RepN
        self.compute_methods = config.repeat.compute_methods
        self._name = config.repeat.name

    def _excete_manual_multiprocess(self) -> None:
        results = multiprocess_evaluate(
            dataset=self.dataset,
            eval_worker=repeat_evaluate_worker,
            num_workers=min(self.num_cpu // 4, 8),
            kwargs={
                "design_batch_size": self.design_batch_size,
                "compute_methods": self.compute_methods,
                "RepN": self.RepN,
            },
        )
        self.to_json(results)

    def execute(self) -> None:
        self._excete_manual_multiprocess()
