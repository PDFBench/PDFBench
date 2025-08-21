import os
from dataclasses import dataclass

from src.datasets import DatasetType


@dataclass
class BasicArguments:
    input_path: str
    design_batch_size: int
    output_dir: str
    dataset_type: DatasetType
    verbose: bool = True
    log_dir: str = "logs"
    visualize: bool = False  # FIXME: NAME
    visual_name: str = "results.csv"
    num_gpu: int = -1
    num_cpu: int = -1

    def init(self):
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
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(os.path.join(self.output_dir, self.log_dir)):
            os.makedirs(self.log_dir)
