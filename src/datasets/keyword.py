import re

from .dataset import BaseDataset


class KeywordDataset(BaseDataset):
    """
    Keyword-guided Dataset for
    [SwissInPro](https://huggingface.co/datasets/Knlife/SwiwwProtIPG)
    and [CAMEOTest](https://huggingface.co/datasets/Knlife/CAMEOTest),
    with keywords from [InterPro](https://www.ebi.ac.uk/interpro), used in PDFBench official experiments.
    """

    @classmethod
    def function(cls, instruction: str) -> str:
        """
        Extract the part of Instruction related to protein function

        :param instruction: Instruction of InterProDataset
        :type instruction: str
        :return: Part of the instruction containing protein function in InterProDataset
        :rtype: str
        """
        keyword = instruction.removesuffix("The designed protein sequence is ")
        keyword = re.search(r":\s*(.*)", keyword[:-2]).group(1)  # type: ignore
        return keyword.strip()
