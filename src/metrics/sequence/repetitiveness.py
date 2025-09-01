import multiprocessing as mp

from tqdm.auto import tqdm

from src.configs.sequence_args import Repeat_Algorithm
from src.datasets import BaseDataset
from src.metrics import BaseMetric
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
    if len(sequence) == n:
        return 0.0

    indeps = set()
    for i in range(len(sequence) - n + 1):
        indeps.add(sequence[i : i + n])

    return max(0.0, 1 - len(indeps) / (len(sequence) - n + 1))


def repeat_evaluate_worker(
    queue: mp.Queue, pid: int, subset: list, **kwargs
) -> None:
    design_batch_size = kwargs.get("design_batch_size")
    verbose = kwargs.get("verbose")
    compute_methods = kwargs.get("compute_methods")
    repn = kwargs.get("RepN")
    if not (
        design_batch_size is not None
        and verbose is not None
        and compute_methods is not None
        and repn is not None
    ):
        raise ValueError(
            f"Invalid kwargs: {design_batch_size}, {verbose}, {compute_methods}, {repn}"
        )

    items = tqdm(
        subset,
        desc="Repetitiveness",
        position=pid + 1,
        ncols=100,
        disable=not verbose or pid != 0,
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

    queue.put(results)


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

    def _evaluate(
        self,
        dataset: BaseDataset,
    ) -> list[dict]:
        return multiprocess_evaluate(
            dataset=dataset,
            eval_worker=repeat_evaluate_worker,
            num_workers=min(self.num_cpu // 4, 8),
            kwargs={
                "design_batch_size": self.design_batch_size,
                "verbose": self.verbose,
                "compute_methods": self.compute_methods,
                "RepN": self.RepN,
            },
        )
