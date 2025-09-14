import json

from src.metrics.metric import EvaluationOutput


def to_json(results: list[EvaluationOutput], output_path: str) -> None:
    final = {}
    for result in results:
        final.update(result.summary)
    for metric in final.keys():
        final[metric] = round(final[metric], 2)
    with open(output_path, "w") as f:
        json.dump(final, f, indent=4)
