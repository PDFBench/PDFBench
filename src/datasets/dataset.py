import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    Base class for datasets
    """

    def __init__(self, path: Union[str, Path]) -> None:
        """
        Load result dataset from JSON file.

        :param str path: Path to the JSON file containing the results in [official format](?)   # TODO: add link
        """
        if type(path) is str:
            assert path.endswith(".json")
        elif type(path) is Path:
            assert path.suffix == ".json"
        else:
            raise TypeError("path must be str or Path")

        with open(path, "r") as f:
            self.data = json.load(f)
            try:
                assert isinstance(self.data, list), "Results must be a list"
                assert len(self.data) > 0, "Results must not be empty"
                assert {"instruction", "reference", "response"}.issubset(
                    self.data[0].keys()
                ), (
                    "Results must contain keys: `instruction`, `reference` and `response`"
                )
            except AssertionError as e:
                raise RuntimeError(
                    "Error in JSON file, please check the above error message and official tutorial in ?"  # TODO: add link
                ) from e

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @classmethod
    @abstractmethod
    def function(cls, instruction: str) -> str:
        """
        Extract the part of `Instruciton` related to **protein function**

        :param str instruciton: Instruction of Dataset
        :return str: Part of the instruction containing protein function in Dataset
        """
        ...
