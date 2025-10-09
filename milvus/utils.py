import subprocess, sys, shlex
from typing import Dict
import yaml
from pathlib import Path
from .conf import Logger


# -- logger settings
logger = Logger(env="dev")


def get_configurations(config_yaml_path: str) -> Dict:
    """
    Gets the compute device

    Params:
        config_yaml_path (str):
            YAML configuration path

    Returns:
        Dict:
            Configurations read from YAML

    """
    with open(config_yaml_path) as f:
        configs = yaml.load(f, Loader=yaml.SafeLoader)
    return configs


def run_command(cmd: str, cwd: Path | str):
    logger.info(f"\n>>> {cmd}")
    res = subprocess.run(shlex.split(cmd), cwd=cwd)
    if res.returncode != 0:
        sys.exit(res.returncode)
    return
