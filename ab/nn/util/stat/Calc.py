from ab.nn.util.stat.DB import *
import random

def patterns_to_configs(config_pattern: str | tuple, random_config_order: bool) -> tuple[str]:
    if not isinstance(config_pattern, tuple):
        config_pattern = (config_pattern,)
    all_configs: list[str] = unique_configs(config_pattern)
    if random_config_order:
        random.shuffle(all_configs)
    else:
        all_configs.sort()
    return tuple(all_configs)
