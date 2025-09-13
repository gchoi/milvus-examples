from typing import Dict
import yaml


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
