import argparse
import os
from dataclasses import dataclass

from src.datasets import DatasetType


@dataclass
class BasicArguments:
    input_path: str
    design_batch_size: int
    output_dir: str
    dataset_type: DatasetType
    log_dir: str = "logs"
    verbose: bool = True
    visualize: bool = False  # FIXME: NAME
    visual_name: str = "results.csv"
    num_gpu: int = -1
    num_cpu: int = -1
    speed_up: bool = False

    def __post_init__(self):
        if self.num_gpu == -1:
            self.num_gpu = len(
                os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            )
        if self.num_cpu == -1:
            cpu_count: int | None = os.cpu_count()
            assert cpu_count is not None, (
                "Python.os cannot detect cpu count of your device, "
                "please set num_cpu manually"
            )
            self.num_cpu = cpu_count
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"input path {self.input_path} not found")
        if self.design_batch_size < 1:
            raise ValueError("design_batch_size must be at least 1")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, self.log_dir), exist_ok=True)
        if not (self.verbose or self.visualize):
            raise ValueError(
                "verbose and visualize cannot both be False, "
                "otherwise the results will be ignored"
            )
        if self.speed_up and self.num_gpu <= 1:
            raise ValueError(
                "speed_up(Accelerate) can only be True when num_gpu > 1"
            )
        parser = argparse.ArgumentParser()
        parser.add_argument("--config_path", required=True)
        args, _ = parser.parse_known_args()
        self.config_path = args.config_path
