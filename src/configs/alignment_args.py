from dataclasses import dataclass


@dataclass
class AlignmentArguments:
    protrek: str
    evollama: str
    llama: str
    interpro_scan_ex: str
    workers_per_interpro_scan: int
