from dataclasses import dataclass


@dataclass
class LaunchArguments:
    metric_cls: str | None = None
