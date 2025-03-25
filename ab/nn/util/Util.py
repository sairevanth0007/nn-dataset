import argparse
import datetime
import gc
import importlib.util
import inspect
import random
import torch

from ab.nn.util.Const import *


def nn_mod(*nms):
    return ".".join(to_nn + nms)


def get_obj_attr(obj, f_name, default=None):
    return getattr(obj, f_name) if hasattr(obj, f_name) else default


def get_attr(mod, f):
    return get_obj_attr(__import__(nn_mod(mod), fromlist=[f]), f)


def min_accuracy(dataset):
    return get_attr(f"loader.{dataset}", 'minimum_accuracy')


def order_configs(configs, random_config_order):
    configs = list(configs)
    if random_config_order:
        random.shuffle(configs)
    else:
        configs.sort()
    return configs


def conf_to_names(c: str) -> tuple[str, ...]:
    return tuple(c.split(config_splitter))


def is_full_config(l: list[str] | tuple[str, ...]):
    return 4 == len(l) and (nn_dir / (l[-1] + '.py')).exists()


def merge_prm(prm: dict, d: dict):
    prm.update(d)
    prm = dict(sorted(prm.items()))
    return prm


def max_batch(binary_power):
    return 2 ** binary_power


def model_stat_dir(config):
    return stat_dir / config_splitter.join(config)


def accuracy_to_time_metric(accuracy, min_accuracy, training_duration) -> float:
    """
    Naive 'accuracy to time' metric for fixed number of training epochs.
    This metric is essential for detecting the fastest accuracy improvements during neural network training.

    """
    d = max(0.0, (accuracy - min_accuracy)) / (training_duration / 1e11)
    print(f"accuracy_to_time_metric {d}")
    return d


def good(result, minimum_accuracy, duration):
    return result > minimum_accuracy * 1.2


def validate_prm(batch_min, batch_max, lr_min, lr_max, momentum_min, momentum_max):
    if batch_min > batch_max: raise Exception(f"min_batch_binary_power {batch_min} > max_batch_binary_power {batch_max}")
    if lr_min > lr_max: raise Exception(f"min_learning_rate {lr_min} > max_learning_rate {lr_max}")
    if momentum_min > momentum_max: raise Exception(f"min_momentum {momentum_min} > max_momentum {momentum_max}")


def format_time(sec):
    return datetime.timedelta(seconds=int(sec))


def release_memory():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def read_py_file_as_string(file_path):
    """
    read_py_file_as_string。

    param:
        file_path (str): path of the file to read.

    Return:
        str: Content of the file.
    """
    try:
        spec = importlib.util.spec_from_file_location("module_name", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        source_code = inspect.getsource(module)
        return source_code
    except Exception as e:
        print(f"error when reading file: {e}")
        return None


def export_model_to_onnx(model, model_name, dummy_input):
    assert isinstance(model, torch.nn.Module)
    hasAdaptivePoolingLayer = False
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveMaxPool2d)):
            if layer.output_size not in [(1, 1), 1, None]:
                hasAdaptivePoolingLayer = True
    onnx_file_path = ab_root_path / 'onnx' / f"{model_name}.onnx"

    if hasAdaptivePoolingLayer:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_file_path,
            input_names=["input"],
            output_names=["output"]
        )
    else:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_file_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size"}
            }
        )
    print(f"Exported {model_name} to ONNX format at {onnx_file_path}")


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
    parser.add_argument('-w', '--workers', type=int, default=default_num_workers,
                        help="Number of data loader workers.")
    parser.add_argument('--pretrained', type=int, choices=[1, 0], default=default_pretrained,
                        help='Control pretrained weights usage: 1 (always use), 0 (never use), or default (let Optuna decide)')

    return parser.parse_args()
