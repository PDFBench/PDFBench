from dataclasses import dataclass


@dataclass
class EvaluationArguments:
    input_path: str
    output_dir: str
    mode: str
    metrics: list[str]
    num_devices: int
    num_workers: int
