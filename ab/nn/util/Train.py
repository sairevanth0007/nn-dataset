import importlib
import math
import time as time_f
import numpy as np
from os.path import join

import torch
from tqdm import tqdm

from ab.nn.util.db.Calc import save_results
from ab.nn.util.Util import nn_mod, merge_prm, get_attr
import ab.nn.util.db.Write as DB_Write


class Train:
    def __init__(self, config, out_shape: tuple, batch: int, model_name, model_stat_dir, task,
                 train_dataset, test_dataset, metric, prm: dict):
        """
        Universal class for training CV, Text Generation and other models.
        :param config: Config (Task, Dataset, Metric, and Model name).
        :param model_name: Neural network model name (e.g., 'ResNet').
        :param model_stat_dir: Path to the model's statistics as a string (e.g., 'ab/nn/stat/img-classification_cifar-10_acc_AlexNet').
        :param task: e.g., 'img-segmentation' to specify the task type.
        :param train_dataset: The dataset used for training the model (e.g., torch.utils.data.Dataset).
        :param test_dataset: The dataset used for evaluating/testing the model (e.g., torch.utils.data.Dataset).
        :param metric: The name of the evaluation metric (e.g., 'acc', 'iou').
        :param out_shape: The shape of output tensor of the model (e.g., number of classes for classification tasks).
        :param batch: Batch size used for both training and evaluation.
        :param prm: dictionary of hyperparameters and their values (e.g., {'lr': 0.11, 'momentum': 0.2})
        """
        self.config = config
        self.model_stat_dir = model_stat_dir
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.out_shape = out_shape
        self.batch = batch
        self.task = task
        self.prm = prm

        self.metric_name = metric
        self.metric_function = self.load_metric_function(metric)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch, shuffle=True, num_workers=2)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch, shuffle=False, num_workers=2)

        for input_tensor, _ in self.train_loader:
            self.in_shape = np.array(input_tensor).shape # Model input tensor shape (e.g., (8, 3, 32, 32) for a batch size 8, RGB image 32x32 px).
            break

        # Load model
        self.model = get_attr(f"nn.{model_name}", "Net")(self.in_shape, out_shape, prm)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self.device = device
        self.model.to(self.device)

    def load_metric_function(self, metric_name):
        """
        Dynamically load the metric function or class based on the metric_name.
        :param metric_name: Name of the metric (e.g., 'acc', 'iou').
        :return: Loaded metric function or initialized class.
        """
        try:
            module = importlib.import_module(nn_mod('metric', metric_name))
            if metric_name == "iou":
                return module.MIoU(self.out_shape)
            else:
                return getattr(module, "compute")
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Metric '{metric_name}' not found. Ensure a corresponding file and function exist.") \
                from e


    def train_n_eval(self, num_epochs):
        """ Training and evaluation """

        duration = time_f.time_ns()
        self.model.train_setup(self.device, self.prm)
        accuracy = None
        for epoch in range(1, num_epochs + 1):
            print(f"epoch {epoch}", flush=True)
            self.model.train()
            self.model.learn(tqdm(self.train_loader))
            accuracy = self.eval(self.test_loader)
            accuracy = 0.0 if math.isnan(accuracy) or math.isinf(accuracy) else accuracy
            prm = merge_prm(self.prm, {'duration': time_f.time_ns() - duration,
                        'accuracy': accuracy})
            save_results(self.config, epoch, join(self.model_stat_dir, f"{epoch}.json"), prm)

        return accuracy

    def eval(self, test_loader):
        """ Evaluation """
        self.model.eval()
        total_correct, total_samples = 0, 0
        if hasattr(self.metric_function, "reset"):  # Check for reset()
            self.metric_function.reset()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                if hasattr(self.metric_function, "update"):  # For mIoU
                    self.metric_function.update(outputs, labels)
                else:  # For accuracy and others
                    correct, total = self.metric_function(outputs, labels)
                    total_correct += correct
                    total_samples += total
        # Metric result
        if hasattr(self.metric_function, "get"):
            result = self.metric_function.get()
        else:
            result = total_correct / total_samples
        return result


def train_new(nn_code, task, dataset, metric, prm):
    # todo: train and if result is appropriate save the code of the new NN model into database
    DB_Write.save_nn(nn_code, task, dataset, metric)
    return None