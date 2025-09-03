import json
from abc import ABC, abstractmethod
from pathlib import Path

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    Base class for datasets
    """

    def __init__(self, path: str | Path, design_batch_size: int) -> None:
        """
        Load result dataset from JSON file.

        :param str path: Path to the JSON file containing the results in [official format](?)   # TODO: add link
        """
        # Path Check
        if type(path) is str:
            assert path.endswith(".json")
        elif type(path) is Path:
            assert path.suffix == ".json"
        else:
            raise TypeError("path must be str or Path")

        # Batch Size
        assert design_batch_size >= 1, "batch_size must be at least 1"
        self._design_batch_size = design_batch_size

        # Load Data
        with open(path, "r") as f:
            self.data = json.load(f)
            try:
                assert isinstance(self.data, list), "Results must be a list"
                assert len(self.data) > 0, "Results must not be empty"
                assert self.support_keys.issubset(self.data[0].keys()), (
                    f"Results must contain keys: {self.support_keys}"
                )
            except AssertionError as e:
                raise RuntimeError(
                    "Error in JSON file, please check the above error message and official tutorial in ?"  # TODO: add link
                ) from e

    @property
    def support_keys(self):
        return {"instruction", "reference"}.union(
            {f"response#{b}" for b in range(1, self.design_batch_size + 1)}
        )

    @property
    def design_batch_size(self):
        return self._design_batch_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key: int | slice) -> dict | list[dict] | list[str]:
        """
        _summary_

        :param int key: _description_
        :raises TypeError: _description_
        :return dict | list[dict] | list[str]: _description_
        """
        if isinstance(key, (int, slice)):
            return self.data[key]
        # elif isinstance(key, str):
        #     # response
        #     assert key in self.support_keys, (
        #         f"key must belong to {self.support_keys}"
        #     )
        #     if key == "response":
        #         return [
        #             {
        #                 f"response#{b}": item[f"response#{b}"]
        #                 for b in range(1, self.batch_size + 1)
        #             }
        #             for item in self.data
        #         ]
        #     else:
        #         return [item[key] for item in self.data]
        else:
            raise TypeError("key must be int, slice")

    @classmethod
    @abstractmethod
    def function(cls, instruction: str) -> str:
        """
        Extract the part of `Instruciton` related to **protein function**

        :param str instruciton: Instruction of Dataset
        :return str: Part of the instruction containing protein function in Dataset
        """
        ...
