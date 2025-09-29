from .dataset import BaseDataset


class CostumedDataset(BaseDataset):
    """
    Costumed dataset for costumed evaluation.
    """
    @classmethod
    def function(cls, instruction: str) -> str:
        """
        Extract the part of Instruction that related to protein function

        :param instruction: Instruction of your costumed dataset
        :type instruction: str
        :return: Part of the instruction containing protein function in your dataset
        :rtype: str
        """
        raise NotImplementedError("Costumed dataset must be implemented by you own to extract the part of Instruction related to protein function")
