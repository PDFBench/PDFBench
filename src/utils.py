import re


def get_text_from_keywords(instruction: str) -> str:
    # We only keep the keyword part of the instruction for evaluation
    keyword = instruction.removesuffix("The designed protein sequence is ")
    keyword = re.search(r":\s*(.*)", keyword[:-2]).group(1)  # type: ignore
    return keyword.strip()


def get_text_from_description(instruction: str) -> str:
    # We only keep the function part of the instruction for evaluation
    function = re.sub(r"^.*?(1\.)", r"\1", instruction)
    function = function.removesuffix("The designed protein sequence is ")
    return function.strip()
