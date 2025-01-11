import argparse
import math
from os.path import exists

from ab.nn.util.Const import *


def nn_mod(*nms):
    return ".".join(to_nn + nms)

def get_attr (mod, f):
    return getattr(__import__(nn_mod(mod), fromlist=[f]), f)


def conf_to_names(c: str) -> list[str]:
    return c.split('_')


def is_full_config(s: str):
    l = conf_to_names(s)
    return 4 == len(l) and exists(join(nn_dir, l[-1] + '.py'))


def merge_prm(prm: dict, d: dict):
    prm.update(d)
    prm = dict(sorted(prm.items()))
    return prm

def max_batch (binary_power):
    return 2 ** binary_power


class CudaOutOfMemory(Exception):
    def __init__(self, batch):
        self.batch_power = int(math.log2(batch))

    def batch_size_power(self):
        return self.batch_power


class ModelException(Exception):
    def __init__(self):
        pass


class AccuracyException(Exception):
    def __init__(self, accuracy, message):
        self.accuracy = accuracy
        self.message = message


def validate_prm(batch_min, batch_max, lr_min, lr_max, momentum_min, momentum_max):
    if batch_min > batch_max: raise Exception(f"min_batch_binary_power {batch_min} > max_batch_binary_power {batch_max}")
    if lr_min > lr_max: raise Exception(f"min_learning_rate {lr_min} > max_learning_rate {lr_max}")
    if momentum_min > momentum_max: raise Exception(f"min_momentum {momentum_min} > max_momentum {momentum_max}")

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=default_config,
                        help="Configuration specifying the model training pipelines. The default value for all configurations.")
    parser.add_argument('-e', '--epochs', type=int, default=default_epochs,
                        help="Numbers of training epochs.")
    parser.add_argument('-t', '--trials', type=int, default=default_trials,
                        help="The total number of Optuna trials the model should have. If negative, its absolute value represents the number of additional trials.")
    parser.add_argument('--min_batch_binary_power', type=int, default=default_min_batch_power,
                        help="Minimum power of two for batch size. E.g., with a value of 0, batch size equals 2**0 = 1.")
    parser.add_argument('-b', '--max_batch_binary_power', type=int, default=default_max_batch_power,
                        help="Maximum power of two for batch size. E.g., with a value of 12, batch size equals 2**12 = 4096.")
    parser.add_argument('--min_learning_rate', type=float, default=default_min_lr,
                        help="Minimum value of learning rate.")
    parser.add_argument('-l', '--max_learning_rate', type=float, default=default_max_lr,
                        help="Maximum value of learning rate.")
    parser.add_argument('--min_momentum', type=float, default=default_min_momentum,
                        help="Minimum value of momentum.")
    parser.add_argument('-m', '--max_momentum', type=float, default=default_max_momentum,
                        help="Maximum value of momentum.")
    parser.add_argument('-f', '--transform', type=str, default=default_transform,
                        help="The transformation algorithm name. If None (default), all available algorithms are used by Optuna.")
    parser.add_argument('-a', '--nn_fail_attempts', type=int, default=default_nn_fail_attempts,
                        help="Number of attempts if the neural network model throws exceptions.")
    parser.add_argument('-r', '--random_config_order', type=bool, default=default_random_config_order,
                        help="If random shuffling of the config list is required.")
    return parser.parse_args()


