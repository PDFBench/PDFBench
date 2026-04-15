import json
import os
import time
import warnings
from random import randint, sample

import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.tokenization import InterProQuantizedTokenizer
from esm.utils.types import FunctionAnnotation
from huggingface_hub import login

warnings.filterwarnings(
    action="ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning,
)


os.environ["ESM_API_KEY"] = "Enter your ESM API key here"
login("Enter your Huggingface token here.")

# Loading ESM3 from huggingface.
model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to("cuda")

# Loading ESM3 from Forge.
# model = client(
#     model="esm3-open",
#     url="https://forge.evolutionaryscale.ai",
#     token="5GL4R8FyNEsF1tDsZsfw5a",
# )


def get_keywords_from_interpro(
    interpro_annotations,
    interpro2keywords=InterProQuantizedTokenizer().interpro2keywords,
):
    keyword_annotations_list = []
    for interpro_annotation in interpro_annotations:
        keywords = interpro2keywords.get(interpro_annotation.label, [])
        keyword_annotations_list.extend(
            [
                FunctionAnnotation(
                    label=keyword,
                    start=interpro_annotation.start,
                    end=interpro_annotation.end,
                )
                for keyword in keywords
            ]
        )
    return keyword_annotations_list


def get_sequence_from_reference(
    sequence: str,
    prefix: int | float,  # type: ignore
    prompt_length: int,
):
    if prompt_length != len(sequence):
        return "".join(["_"] * prompt_length)
    if isinstance(prefix, float):
        if 0.0 < prefix < 1.0:
            prefix: int = int(prompt_length * prefix)
        else:
            raise ValueError("Prefix must be int or float in [0.0, 1.0]")
    else:
        prefix: int = prefix
    return sequence[:prefix] + "".join(["_"] * (prompt_length - prefix))


def annotation_check(annotations: list[FunctionAnnotation]):
    for idx in range(len(annotations)):
        start, end = annotations[idx].start, annotations[idx].end
        if end > 512 or start > 512:
            p, q = randint(256, 512), randint(256, 512)
            start = min(p, q)
            end = max(p, q)
        annotations[idx] = FunctionAnnotation(
            label=annotations[idx].label, start=start, end=end
        )

    maximum = max([keyword.end for keyword in annotations])
    return annotations, randint(maximum, 512)


def keyword2sequence(
    annotations: list[FunctionAnnotation],
    prompt_length: int | None = None,
    reference: str = "",
    prefix: int | float = 0,
) -> str:
    print(f"prompt_length({prompt_length}) set the same as ground_truth")
    function_annotations = get_keywords_from_interpro(annotations)
    sequence_prompt: str = get_sequence_from_reference(
        reference, prefix, prompt_length
    )
    protein_prompt = ESMProtein(
        sequence=sequence_prompt, function_annotations=function_annotations
    )

    func2seq_generation_config = GenerationConfig(
        track="sequence",
        num_steps=prompt_length // 8,
        temperature=0.5,
    )
    sequence_generation: ESMProtein = model.generate(
        protein_prompt, func2seq_generation_config
    )  # type: ignore

    assert sequence_generation.sequence is not None
    return sequence_generation.sequence


def main(
    input_path: str,
    output_path: str,
    num_keyword: int | None = None,
    prefix: int | float = 0,
):
    with open(input_path) as f:
        data = json.load(f)

    if num_keyword:
        _data = []
        for item in data:
            if num_keyword < len(item["Keywords"]):
                item["Keywords"] = sample(item["Keywords"], num_keyword)
            _data.append(item)
        data = _data

    for idx in range(len(data)):
        # reference should be the same as sequence
        reference = data[idx]["sequence"]

        # extract keywords from
        keywords = data[idx]["InterPro"]
        keywords = [
            FunctionAnnotation(
                label=keyword["InterPro-ID"],
                start=keyword["Beg"],
                end=keyword["End"],
            )
            for keyword in keywords
        ]

        # select prompt length
        prompt_length = min(len(data[idx]["sequence"]), 2048)

        # generate
        print(f"[{idx + 1} / {len(data)}]", end="\t")
        data[idx]["response#1"] = keyword2sequence(
            keywords, prompt_length, reference, prefix
        )
        data[idx]["response#2"] = keyword2sequence(
            keywords, prompt_length, reference, prefix
        )
        data[idx]["response#3"] = keyword2sequence(
            keywords, prompt_length, reference, prefix
        )
        data[idx]["reference"] = reference
        data[idx].pop("sequence")

        torch.cuda.empty_cache()

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def test_keyword2sequence():
    interpro_annotations = [
        FunctionAnnotation(label="IPR050145", start=1, end=142),
        FunctionAnnotation(label="IPR002048", start=4, end=75),
        FunctionAnnotation(label="IPR002048", start=77, end=144),
        FunctionAnnotation(label="IPR011992", start=1, end=143),
        FunctionAnnotation(label="IPR018247", start=17, end=29),
        FunctionAnnotation(label="IPR018247", start=53, end=65),
        FunctionAnnotation(label="IPR018247", start=90, end=102),
        FunctionAnnotation(label="IPR018247", start=126, end=138),
    ]
    times = 0
    while times < 3:
        try:
            sequence = keyword2sequence(interpro_annotations)
            print(sequence)
        except Exception as e:
            print(f"Error: {e}")
            continue
        times += 1
        print(f"NO.{times} Generation- {sequence[:30]}")
        time.sleep(15)


if __name__ == "__main__":
    DEBUG = True
    if DEBUG:
        test_keyword2sequence()
    else:
        from fire import Fire

        Fire(main)
