# This file is copied from [EvoLlama](https://github.com/sornkL/EvoLlama)
# Original license: MIT License
import argparse

from dynaconf import Dynaconf

config_filename = "default.yaml"
# config_filename = 'map.yaml'
# config_filename = 'pharm.yaml'


def load_config(default_config_file: str = config_filename) -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config_file)
    args, _ = parser.parse_known_args()
    return args.config


config = Dynaconf(settings_files=[load_config()])
