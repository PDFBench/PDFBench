import json
import os

import matplotlib.pyplot as plt


def boxplot(evaluation_dir: str, key: str):
    json_files = []
    for root, _, files in os.walk(evaluation_dir):
        for file in files:
            if file == "evaluation.json":
                json_files.append(os.path.join(root, file))
    json_files = sorted(json_files, key=lambda x: x.split("/")[-2])

    data = []
    labels = []
    for file in json_files:
        category = file.split("/")[-2].split(".")[0]

        with open(file, "r") as f:
            json_data = json.load(f)

        values = [sample[key] for sample in json_data if key in sample]
        data.append(values)
        labels.append(category)

        print(f"The average {key} of {category} == {sum(values) / len(values)}")

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        data,
        labels=labels,  # type: ignore
        patch_artist=True,
        boxprops=dict(facecolor="#73aedf", alpha=0.7),
    )
    plt.title(key.upper(), fontsize=16)
    plt.ylabel(key.upper(), fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(evaluation_dir, f"boxplot_{key}.pdf"), format="pdf"
    )
