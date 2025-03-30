import sys
import yaml
from pathlib import Path
from tqdm import tqdm as std_tqdm
from tqdm.notebook import tqdm as nb_tqdm


def merge_dicts(dict_list):
    merged = {}
    for d in dict_list:
        for key, value in d.items():
            merged.setdefault(key, []).append(value)  # Collect values as lists
    return merged


def get_tqdm():
    if 'ipykernel' in sys.modules:
        return nb_tqdm  # Jupyter notebook or Lab
    else:
        return std_tqdm


def load_params(config_path: Path = Path("../params.yaml")):
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)

    params["qdrant"]["model_name"] = params["embedding"]["model_name"]

    return params
