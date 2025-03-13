import json
from os.path import exists
from pathlib import Path

import ab.nn.util.db.Read as DB_Read
from ab.nn.util.Util import conf_to_names, order_configs


def patterns_to_configs(config_pattern: str | tuple, random_config_order: bool) -> tuple[tuple[str, ...], ...]:
    """
    Generate unique configurations based on the input pattern(s).
    :param config_pattern: A string or tuple of configuration patterns.
    :param random_config_order: Whether to shuffle the configurations randomly.
    :return: A tuple of unique configurations.
    """
    if not isinstance(config_pattern, (tuple, list)):
        config_pattern = [config_pattern]
    config_pattern = [conf_to_names(config) for config in config_pattern]
    configs = DB_Read.unique_configs(config_pattern)
    configs = order_configs(configs, random_config_order)
    return tuple(configs)


def save_results(config_ext: tuple[str, str, str, str, int], model_stat_file: str, prm: dict):
    trials_dict_all = [prm]

    if exists(model_stat_file):
        with open(model_stat_file, 'r') as f:
            previous_trials = json.load(f)
            trials_dict_all = previous_trials + trials_dict_all

    trials_dict_all = sorted(trials_dict_all, key=lambda x: x['accuracy'], reverse=True)
    # Save trials.json
    Path(model_stat_file).parent.absolute().mkdir(exist_ok=True)
    with open(model_stat_file, 'w') as f:
        json.dump(trials_dict_all, f, indent=4)

    print(f"Trial (accuracy {prm['accuracy']}) for ({', '.join([str(o) for o in config_ext])}) saved at {model_stat_file}")
