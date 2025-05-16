import json
import multiprocessing as mp
import os
import re
import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer, logging

from src.eval.EvoLlama.src.infer.infer import infer, init_evo_llama
from src.eval.foldability import get_md5_sequence
from src.eval.ProTrek.model.ProTrek.protrek_trimodal_model import (
    ProTrekTrimodalModel,
)
from src.eval.ProTrek.utils.foldseek_util import get_struc_seq

logging.set_verbosity_error()
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="TypedStorage is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`clean_up_tokenization_spaces` was not set.*",
)


def get_cameo_text(instruction: str) -> str:
    # We only keep the keyword part of the instruction for evaluation
    keyword = instruction.removesuffix("The designed protein sequence is ")
    keyword = re.search(r":\s*(.*)", keyword[:-2]).group(1)

    return keyword.strip()


def get_molinst_text(instruction: str) -> str:
    # We only keep the function part of the instruction for evaluation
    function = re.sub(r"^.*?(1\.)", r"\1", instruction)
    function = function.removesuffix("The designed protein sequence is ")

    return function.strip()


def get_embedding(
    model: BertModel, tokenizer: BertTokenizer, texts: List[str]
) -> torch.Tensor:
    def mean_pooling(output, mask):
        embeddings = output[
            0
        ]  # First element of model_output contains all token embeddings
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(
            mask.sum(1), min=1e-9
        )

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    return mean_pooling(outputs, inputs["attention_mask"])


def _main_protrek_score(
    uid: int,
    model_path: str,
    use_structure: bool,
    use_sequence: bool,
    subset: List[Tuple[str, str, str]],
):
    config = {
        "protein_config": os.path.join(model_path, "esm2_t33_650M_UR50D"),
        "text_config": os.path.join(
            model_path, "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        ),
        "structure_config": os.path.join(model_path, "foldseek_t30_150M"),
        "load_protein_pretrained": False,
        "load_text_pretrained": False,
        "from_checkpoint": os.path.join(model_path, "ProTrek_650M_UniRef50.pt"),
    }
    model = ProTrekTrimodalModel(**config).eval().to(f"cuda:{uid}")

    ret = []
    with torch.no_grad():
        for data in tqdm(
            subset,
            desc=f"Process {uid + 1} - Protrek Score",
            position=uid + 1,
            ncols=100,
        ):
            text, sequence, structure = data
            text_embedding = model.get_text_repr([text])
            if use_sequence:
                sequence_embedding = model.get_protein_repr([sequence])
                seq_protrek_score = torch.nn.functional.cosine_similarity(
                    sequence_embedding, text_embedding
                ).item()
            else:
                seq_protrek_score = None
            if use_structure and os.path.exists(structure):
                foldseek_structure = get_struc_seq(
                    "src/eval/ProTrek/bin/foldseek", structure, ["A"]
                )["A"][1].lower()
                structure_embedding = model.get_structure_repr(
                    [foldseek_structure]
                )
                struct_protrek_score = torch.nn.functional.cosine_similarity(
                    structure_embedding, text_embedding
                ).item()
            else:
                struct_protrek_score = None
            ret.append(
                {
                    "sequence": sequence,
                    "text": text,
                    "protrek_score (seq)": seq_protrek_score,
                    "protrek_score (struct)": struct_protrek_score,
                }
            )

    return ret


def _main_evollama_score(
    uid: int,
    model_path: str,
    llm_path: str,
    use_structure: bool,
    use_sequence: bool,
    subset: List[Tuple[str, str, str]],
):
    prompt = "The function of the protein is:\n"
    prompt = [prompt]
    model = (
        init_evo_llama(
            structure_encoder_path=os.path.join(
                model_path, "structure_encoder_weights.bin"
            ),
            structure_encoder_name="ProteinMPNN",
            sequence_encoder_path=os.path.join(model_path, "sequence_encoder"),
            llm_path=llm_path,
            projection_path=os.path.join(model_path, "projection_weights.bin"),
            projection_fusion=use_structure and use_sequence,
            is_inference=True,
            llm_embedding_dim=3072,
        )
        .eval()
        .to(f"cuda:{uid}")  # type: ignore
    )

    embedding_model = BertModel.from_pretrained(
        "/home/nwliu/data/pretrain/pubmedbert-base-embeddings"
    ).to(f"cuda:{uid}")  # type: ignore
    embedding_tokenizer = BertTokenizer.from_pretrained(
        "/home/nwliu/data/pretrain/pubmedbert-base-embeddings"
    )
    ret = []
    with torch.no_grad():
        for data in tqdm(
            subset,
            desc=f"Process {uid} - Evollama Score",
            position=uid + 1,
            ncols=100,
        ):
            text, sequence, structure = data
            input_structure = [[structure]] if use_structure else None
            input_sequence = [[sequence]] if use_sequence else None
            response = infer(
                model,
                input_structure,  # type: ignore
                input_sequence,  # type: ignore
                prompt,
            )[0]
            embeddings = get_embedding(
                embedding_model,  # type: ignore
                embedding_tokenizer,
                [response, text],  # type: ignore
            )
            ret.append(
                {
                    "sequence": sequence,
                    "text": text,
                    "predictions": response,
                    "evollama_score": nn.functional.cosine_similarity(
                        embeddings[0], embeddings[1], dim=0
                    ).item(),
                }
            )

    return ret


def _main(
    uid: int,
    queue: mp.Queue,
    use_structure: bool,
    use_sequence: bool,
    subset: List[Tuple[str, str, str]],
    protrek_path: str,
    evollama_path: str,
    llm_path: str,
):
    if protrek_path is not None:
        protrek_scores = _main_protrek_score(
            uid, protrek_path, use_structure, use_sequence, subset
        )

    if evollama_path is not None and llm_path is not None:
        evollama_scores = _main_evollama_score(
            uid, evollama_path, llm_path, use_structure, use_sequence, subset
        )

    if (
        protrek_path is not None
        and evollama_path is not None
        and llm_path is not None
    ):
        ret = []
        for protrek_score, evollama_score in zip(
            protrek_scores, evollama_scores
        ):
            if protrek_score["sequence"] == evollama_score["sequence"]:
                protrek_score.update(evollama_score)
                ret.append(protrek_score)
        queue.put(ret)
    else:
        queue.put(
            protrek_scores if protrek_path is not None else evollama_scores
        )


def main(
    num_workers: int,
    use_structure: bool,
    use_sequence: bool,
    task: str,
    pdb_dir: str,
    sequence_file: str,
    evaluation_file: str,
    protrek_path: Optional[str] = None,
    evollama_path: Optional[str] = None,
    llm_path: Optional[str] = None,
    evaluation_dir: Optional[str] = None,
    save_plot: bool = False,
):
    assert sequence_file and evaluation_file

    if not os.path.exists(evaluation_file):
        mp.set_start_method("spawn", force=True)

        with open(sequence_file, "r") as f:
            data = json.load(f)

        get_text = get_molinst_text if task == "molinst" else get_cameo_text
        data = [
            (
                get_text(item["instruction"]),
                item["response"],
                os.path.join(
                    pdb_dir, f"{get_md5_sequence(item['response'])}.pdb"
                ),
            )
            for item in data
        ]

        queue = mp.Queue()
        processes = []
        results = []
        for i in range(num_workers):
            begin_idx = i * (len(data) // num_workers)
            end_idx = (
                (i + 1) * (len(data) // num_workers)
                if i != num_workers - 1
                else len(data)
            )
            subset = data[begin_idx:end_idx]
            p = mp.Process(
                target=_main,
                args=(
                    i,
                    queue,
                    use_structure,
                    use_sequence,
                    subset,
                    protrek_path,
                    evollama_path,
                    llm_path,
                ),
            )
            p.start()
            processes.append(p)

        for _ in processes:
            ret = queue.get()
            results.extend(ret)

        for p in processes:
            p.join()

        with open(evaluation_file, "w") as f:
            json.dump(results, f, indent=4)
    else:
        print("Load processed evaluation file")
        with open(evaluation_file, "r") as f:
            results = json.load(f)

    if protrek_path is not None:
        if use_sequence:
            print(
                {
                    "mean Protrek Score (seq)": np.mean(
                        [item["protrek_score (seq)"] for item in results]
                    )
                }
            )
        if use_structure:
            print(
                {
                    "mean Protrek Score (struct)": np.mean(
                        [item["protrek_score (struct)"] for item in results]
                    )
                }
            )
    if evollama_path is not None and llm_path is not None:
        print(
            {
                "EvoLlama Score": np.mean(
                    [item["evollama_score"] for item in results]
                )
            }
        )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
