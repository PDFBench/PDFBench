import json
import os

import numpy as np
import pandas as pd

EPOCH_METRIC2NAME = {
    "perplexity_protgpt2": "perplexity",
    "perplexity_progen": "perplexity",
    "repetitiveness": "repetitiveness",
    "repeat_2": "repetitiveness",
    "repeat_5": "repetitiveness",
    "novelty": "novelty",
    "bertscore_f1": "bertlike",
    "bertscore_precision": "bertlike",
    "bertscore_recall": "bertlike",
    "plddt": "foldability",
    "pae": "foldability",
    "protrek_score (seq)": "language_alignment",
    "evollama_score": "language_alignment",
    "retrieval_accuracy_4": "retrieval_accuracy",
    "retrieval_accuracy_10": "retrieval_accuracy",
    "retrieval_accuracy_20": "retrieval_accuracy",
    "soft_retrieval_accuracy_4": "retrieval_accuracy",
    "soft_retrieval_accuracy_10": "retrieval_accuracy",
    "soft_retrieval_accuracy_20": "retrieval_accuracy",
    "hard_retrieval_accuracy_4": "retrieval_accuracy",
    "hard_retrieval_accuracy_10": "retrieval_accuracy",
    "hard_retrieval_accuracy_20": "retrieval_accuracy",
    "keyword_recovery": "keyword_recovery",
    "tm_score": "tm_score",
    "identity": "identity",
}
BATCH_METRIC2NAME = {"diversity": "diversity"}
FINAL_METRIC = list(EPOCH_METRIC2NAME.keys()) + list(BATCH_METRIC2NAME.keys())
FINAL_METRIC.insert(FINAL_METRIC.index("plddt") + 1, "plddt#N")
FINAL_METRIC.insert(FINAL_METRIC.index("pae") + 1, "pae#N")


def _compute_batch_metric(metric: str, model_dir: str):
    metric_path = os.path.join(model_dir, f"{BATCH_METRIC2NAME[metric]}.json")
    if not os.path.exists(metric_path):
        return {metric: pd.NA}
    with open(metric_path) as f:
        values = [
            sample[metric]
            for sample in json.load(f)
            if metric in sample and not np.isnan(sample[metric])
        ]

    if metric == "diversity":
        return {metric: round(np.mean(values) * 100, 2)}
    else:
        raise NotImplementedError


def compute_batch_metrics(model_dir: str, epoch_results: list) -> dict:
    global EPOCH_METRIC2NAME, BATCH_METRIC2NAME

    if not epoch_results:
        return {}

    batch_res = {}
    for metric in epoch_results[0].keys():
        if metric == "name":
            continue

        values = [epoch_res[metric] for epoch_res in epoch_results]
        batch_mean = (
            pd.NA
            if any([pd.isna(value) for value in values])
            else np.mean(values)
        )
        batch_std = (
            pd.NA
            if any([pd.isna(value) for value in values])
            else np.std(values, axis=0, ddof=1)
        )
        if pd.isna(batch_mean) and pd.isna(batch_std):
            batch_res.update({metric: pd.NA})
        else:
            batch_res.update(
                {metric: rf"{batch_mean:.2f} $$\pm$$ {batch_std:.2f}"}
            )

    for metric in BATCH_METRIC2NAME.keys():
        batch_res.update(_compute_batch_metric(metric, model_dir))

    return batch_res


def _compute_epoch_metric(metric: str, epoch_dir: str):
    metric_path = os.path.join(epoch_dir, f"{EPOCH_METRIC2NAME[metric]}.json")
    if not os.path.exists(metric_path):  # not finished
        res = {metric: pd.NA}
        if metric == "plddt":
            res.update({"plddt#N": pd.NA})
        elif metric == "pae":
            res.update({"pae#N": pd.NA})
        return res
    else:  # finished
        with open(metric_path) as f:
            values = [
                sample[metric]
                for sample in json.load(f)
                if metric in sample and not np.isnan(sample[metric])
            ]
        if metric == "plddt":
            res = {
                "plddt": round(np.mean(values) * 100, 2),
                "plddt#N": round(np.mean(np.array(values) > 0.7) * 100, 2),
            }
        elif metric == "pae":
            res = {
                "pae": round(np.mean(values), 2),
                "pae#N": round(np.mean(np.array(values) < 10.0) * 100, 2),
            }
        elif metric in [
            "repetitiveness",
            "repeat_2",
            "repeat_5",
            "novelty",
            "bertscore_f1",
            "bertscore_precision",
            "bertscore_recall",
            "protrek_score (seq)",
            "evollama_score",
            "retrieval_accuracy_4",
            "retrieval_accuracy_10",
            "retrieval_accuracy_20",
            "soft_retrieval_accuracy_4",
            "soft_retrieval_accuracy_10",
            "soft_retrieval_accuracy_20",
            "hard_retrieval_accuracy_4",
            "hard_retrieval_accuracy_10",
            "hard_retrieval_accuracy_20",
            "keyword_recovery",
            "tm_score",
            "identity",
        ]:
            res = {metric: round(np.mean(values) * 100, 2)}
        elif metric in ["perplexity_protgpt2", "perplexity_progen"]:
            res = {metric: round(np.mean(values), 2)}
        else:
            print(f"Metric {metric} not implemented")
            raise NotImplementedError
        return res


def compute_epoch_metrics(epoch_dir: str) -> dict:
    global EPOCH_METRIC2NAME
    epoch_res = {}
    for metric in EPOCH_METRIC2NAME.keys():
        epoch_res.update(_compute_epoch_metric(metric, epoch_dir))
    return epoch_res


def main(evaluation_dir: str) -> None:
    global EPOCH_METRIC2NAME, BATCH_METRIC2NAME
    result = []
    for file in os.listdir(evaluation_dir):
        if not os.path.isdir(os.path.join(evaluation_dir, file)):
            continue

        model_dir = os.path.join(evaluation_dir, file)

        if file == "ground_truth":
            res = compute_epoch_metrics(model_dir)
            res["name"] = file
            result.append(res)
            continue

        epoch_results = []
        epoch = 1
        while True:
            epoch_dir = os.path.join(model_dir, str(epoch))
            if not os.path.exists(epoch_dir):
                break
            epoch_result = compute_epoch_metrics(epoch_dir)
            epoch_result["name"] = f"{file}#{epoch}"
            epoch_results.append(epoch_result)
            epoch += 1
        batch_result = compute_batch_metrics(model_dir, epoch_results)
        batch_result["name"] = file
        result.append(batch_result)
        result.extend(epoch_results)

    pd.DataFrame(result, columns=["name"] + FINAL_METRIC).to_csv(
        os.path.join(evaluation_dir, "metrics.csv"), index=False
    )
    print("Output metrics to", os.path.join(evaluation_dir, "metrics.csv"))


def test():
    eval_dir = "/home/jhkuang/data/cache/dynamsa/eval/test_molinst_denovo"
    main(eval_dir)


if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        print("Debuging Metric")
        test()
    else:
        import fire

        fire.Fire(main)
