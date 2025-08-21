import os
from dataclasses import dataclass

from src.datasets import DatasetType


@dataclass
class BasicArguments:
    input_path: str
    design_batch_size: int
    output_dir: str
    dataset_type: DatasetType
    verbose: bool
    log_dir: str = "logs"
    visualize: bool = False  # FIXME: NAME
    visual_name: str = "results.csv"
    num_gpu: int = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
    num_cpu: int = -1
