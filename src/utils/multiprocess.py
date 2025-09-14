import multiprocessing as mp
from typing import Any, Callable

from src.datasets import BaseDataset


def multiprocess_evaluate(
    dataset: BaseDataset | list,
    eval_worker: Callable,
    num_workers: int,
    kwargs,
) -> list[Any]:
    """
    _summary_

    :param BaseDataset dataset: _description_
    :param Callable eval_worker: _description_
    :param int num_workers: _description_
    :param Optional[list] args: _description_, defaults to None
    :return list[Any]: _description_
    """
    queue: mp.Queue = mp.Queue()
    procs: list = []
    for pid in range(num_workers):
        piece: int = len(dataset) // num_workers
        beg_idx: int = pid * piece
        end_idx: int = (
            (pid + 1) * piece if pid != num_workers - 1 else len(dataset)
        )
        subset = dataset[beg_idx:end_idx]

        proc = mp.Process(
            target=eval_worker,
            args=(queue, num_workers - pid - 1, subset),  # reverse the pid
            kwargs=kwargs,
        )
        proc.start()
        procs.append(proc)

    results = [[] for _ in range(num_workers)]
    for _ in range(num_workers):
        pid, sub_results = queue.get()
        results[pid] = sub_results
    results.reverse()  # match the pid reversion
    results = [element for sublist in results for element in sublist]

    for proc in procs:
        proc.join()
    return results
