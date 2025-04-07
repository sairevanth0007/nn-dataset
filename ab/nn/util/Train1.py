import importlib
import os
import sys
import tempfile
import time as time
from os.path import join
from typing import Union

import math
import numpy as np
import torch
from torch.cuda import OutOfMemoryError

import ab.nn.util.CodeEval as codeEvaluator
import ab.nn.util.db.Write as DB_Write
from ab.nn.util.Classes import DataRoll
from ab.nn.util.Exception import *
from ab.nn.util.Loader import load_dataset
from ab.nn.util.Util import *
from ab.nn.util.db.Calc import save_results
from ab.nn.util.db.Read import supported_transformers


def optuna_objective(trial, config, num_workers, min_lr, max_lr, min_momentum, max_momentum,
                     min_batch_binary_power, max_batch_binary_power_local, transform, fail_iterations, n_epochs, pretrained):
    print(f"[ERROR] Exception caught in optuna_objective: ")  # placeholder in case of exception
    task, dataset_name, metric, nn = config
    try:
        # Load model
        s_prm: set = get_attr(f"nn.{nn}", "supported_hyperparameters")()
        # Suggest hyperparameters
        prms = {}
        for prm in s_prm:
            match prm:
                case 'lr':
                    prms[prm] = trial.suggest_float(prm, min_lr, max_lr, log=True)
                case 'momentum':
                    prms[prm] = trial.suggest_float(prm, min_momentum, max_momentum)
                case 'dropout':
                    prms[prm] = trial.suggest_float(prm, 0.0, 0.5)
                case 'pretrained':
                    prms[prm] = float(pretrained if pretrained else trial.suggest_categorical(prm, [0, 1]))
                case _:
                    prms[prm] = trial.suggest_float(prm, 0.0, 1.0)
        batch = trial.suggest_categorical('batch', [max_batch(x) for x in range(min_batch_binary_power, max_batch_binary_power_local + 1)])
        transform_name = trial.suggest_categorical('transform', transform if transform else supported_transformers())
        prms = merge_prm(prms, {'batch': batch, 'transform': transform_name})
        prm_str = ', '.join([f"{k}: {v}" for k, v in prms.items()])
        # Load dataset
        out_shape, minimum_accuracy, train_set, test_set = load_dataset(task, dataset_name, transform_name)
        
        # Initialize training with hyperparameters
        if task == 'txt-generation':
            if nn.lower() == 'rnn':
                from ab.nn.nn.RNN import Net as RNNNet
                model = RNNNet(1, 256, len(train_set.chars), batch)
            elif nn.lower() == 'lstm':
                from ab.nn.nn.LSTM import Net as LSTMNet
                model = LSTMNet(1, 256, len(train_set.chars), batch, num_layers=2)
            else:
                raise ValueError(f"Unsupported text generation model: {nn}")
        return Train(config, out_shape, minimum_accuracy, batch, f"nn.{nn}", task, train_set, test_set, metric,
                     num_workers, prms).train_n_eval(n_epochs)

    except Exception as e:
        accuracy_duration = (0.0, 1)
        if isinstance(e, OutOfMemoryError):
            if max_batch_binary_power_local <= min_batch_binary_power:
                return accuracy_duration
            else:
                raise CudaOutOfMemory(batch)
        elif isinstance(e, AccuracyException):
            print(f"[WARN] AccuracyException: {e.message}")
            return e.accuracy, e.duration
        elif isinstance(e, LearnTimeException):
            print(f"[WARN] LearnTimeException: Estimated training time: {format_time(e.estimated_training_time)}, limit: {format_time(e.max_learn_seconds)}.")
            return (e.max_learn_seconds / e.estimated_training_time ) / 1e5, e.duration
        else:
            print(f"[ERROR] '{nn}': failed to train. Error: {e}")
            if fail_iterations < 0:
                return accuracy_duration
            else:
                raise NNException() from e


class Train:
    def __init__(self,
                 config: tuple[str, str, str, str],
                 out_shape: tuple,
                 minimum_accuracy: float,
                 batch: int,
                 model_name,
                 task,
                 train_dataset,
                 test_dataset,
                 metric,
                 num_workers,
                 prm: dict,
                 save_to_db=True,
                 is_code=False,
                 save_path: Union[str, None] = None):
        """
        Universal class for training CV, Text Generation and other models.
        :param config: Tuple of names (Task, Dataset, Metric, Model).
        :param out_shape: Shape of output tensor of the model (e.g., number of classes for classification tasks).
        :param batch: Batch size used for both training and evaluation.
        :param minimum_accuracy: Expected average value for accuracy provided by the untrained NN model due to random output generation. This value is essential for excluding NN models without accuracy gains.
        :param nn_module: Neural network model name (e.g., 'ab.nn.nn.ResNet', 'out.tmp.').
        :param task: e.g., 'img-segmentation' to specify the task type.
        :param train_dataset: Dataset used for training the model (e.g., torch.utils.data.Dataset).
        :param test_dataset: Dataset used for evaluating/testing the model (e.g., torch.utils.data.Dataset).
        :param metric: Name of the evaluation metric (e.g., 'acc', 'iou').
        :param prm: Dictionary of hyperparameters and their values (e.g., {'lr': 0.11, 'momentum': 0.2})
        :param is_code: Whether `config.model` is `nn_code` or `nn`
        :param save_path: Path to save the statistics, set to `None` to use the default
        """
        self.config = config
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.minimum_accuracy = minimum_accuracy

        self.out_shape = out_shape
        self.batch = batch
        self.task = task
        self.prm = prm

        self.metric_name = metric
        self.metric_function = self.load_metric_function(metric)
        self.save_to_db = save_to_db
        self.is_code = is_code
        self.save_path = save_path

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch,
            shuffle=True,
            num_workers=get_obj_attr(self.train_dataset, 'num_workers', default=num_workers),
            collate_fn=lambda batch: self.train_dataset.__class__.collate_fn(batch, self.train_dataset.word2idx)
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch,
            shuffle=False,
            num_workers=get_obj_attr(self.test_dataset, 'num_workers', default=num_workers),
            collate_fn=lambda batch: self.test_dataset.__class__.collate_fn(batch, self.test_dataset.word2idx)
        )

        for input_tensor, _ in self.train_loader:
            self.in_shape = np.array(input_tensor).shape
            break

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self.device = device

        # Load model
        model_net = get_attr(model_name, "Net")
        self.model = model_net(self.in_shape, out_shape, prm, self.device)
        self.model.to(self.device)

    def load_metric_function(self, metric_name):
        """
        Dynamically load the metric function or class based on the metric_name.
        :param metric_name: Name of the metric (e.g., 'acc', 'iou', 'bleu').
        :return: Loaded metric function or initialized class.
        """        
        try:
            module = importlib.import_module(nn_mod('metric', metric_name))
            
            return module.create_metric(self.out_shape)
        
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Metric '{metric_name}' not found. Ensure a corresponding file and function exist. Ensure the metric module has create_metric()") from e

    def train_n_eval(self, num_epochs):
        """ Training and evaluation """
        
        start_time = time.time_ns()
        self.model.train_setup(self.prm)

        accuracy_to_time = 0.0
        duration = sys.maxsize
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            self.model.learn(DataRoll(self.train_loader))
            
            accuracy = self.eval(self.test_loader)
            accuracy = 0.0 if math.isnan(accuracy) or math.isinf(accuracy) else accuracy
            duration = time.time_ns() - start_time
            # The accuracy-to-time metric is not stored in the database as it can change over time and can be quickly calculated from saved values.
            accuracy_to_time = accuracy_to_time_metric(accuracy, self.minimum_accuracy, duration)
            if not good(accuracy, self.minimum_accuracy, duration):
                msg = (f"Accuracy is too low: {accuracy}. The minimum accepted accuracy for the '{self.config[1]}' "
                       f"dataset is {self.minimum_accuracy}.")
                print(f"[ERROR] {msg}")
                raise AccuracyException(accuracy, duration, msg)

            prm = merge_prm(self.prm, {'duration': duration, 'accuracy': accuracy, 'uid': DB_Write.uuid4()})
            if self.save_to_db:
                if self.is_code:  # We don't want the filename contain full codes
                    if self.save_path is None:
                        print(f"[WARN] parameter `save_Path` is None, stats will not be saved into a file.")
                    else:
                        save_results(self.config + (epoch,), join(self.save_path, f"{epoch}.json"), prm)
                else:  # Legacy save result codes in file
                    if self.save_path is None:
                        self.save_path = model_stat_dir(self.config)
                    save_results(self.config + (epoch,), join(self.save_path, f"{epoch}.json"), prm)
                    DB_Write.save_results(self.config + (epoch,), prm)  # Separated from Calc.save_results()
        return accuracy_to_time, duration

    def eval(self, test_loader):
        """Evaluation with standardized metric interface"""
        self.model.eval()
        
        # Reset the metric at the start of evaluation
        self.metric_function.reset()

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if inputs.dim() == 5:
                    inputs = inputs.squeeze(1)
                labels = labels[:, 0, :]
                hidden_state = self.model.rnn.init_zero_hidden(batch=inputs.size(0), device=self.device)
                outputs, _ = self.model(inputs, labels, hidden_state)
                
                # Call the metric - all metrics now use the same interface
                self.metric_function(outputs, labels)
                if batch_idx % 10 == 0:
                    pass

        # Get the final result from the metric
        return self.metric_function.result()


def train_new(nn_code, task, dataset, metric, prm, save_to_db=True, prefix: Union[str, None] = None, save_path: Union[str, None] = None):
    """
    train the model with the given code and hyperparameters and evaluate it.

    parameter:
        nn_code (str): Code of the model
        task (str): Task type
        dataset (str): Name of the dataset
        metric (str): Evaluation metric
        prm (dict): Hyperparameters, e.g., 'lr', 'momentum', 'batch', 'epoch', 'dropout'
        prefix (str|None): Prefix of the model, set to None if is unknown.
        save_path (str|None): Path to save the statistics, or None to not save.
        export_onnx (bool): Export model and its weights into ONNX file.
    return:
        (str, float): Name of the model and the accuracy
    """
    if prefix is None:
        name = None
    else:
        name = prefix + "-" + DB_Write.uuid4()  # Create temporal name for processing

    spec = importlib.util.find_spec("ab.nn.tmp")
    dir_path = os.path.dirname(spec.origin)

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=True, dir=dir_path) as temp_file:
        temp_file_path = temp_file.name
        temp_filename = os.path.basename(temp_file.name).replace(".py", "")
        temp_file.write(nn_code)
        try:
            temp_file.seek(0)
            print("[ERROR] Evaluating code with codeEvaluator.evaluate_single_file")  # placeholder
            res = codeEvaluator.evaluate_single_file(temp_file_path)
            print("[ERROR] Module import placeholder")  # placeholder
            spec = importlib.util.spec_from_file_location(f"ab.nn.tmp.{temp_filename}", temp_file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"ab.nn.tmp.{temp_filename}"] = module
            spec.loader.exec_module(module)
            chosen_transform = prm.get('transform', None)
            # load dataset
            out_shape, minimum_accuracy, train_set, test_set = load_dataset(task, dataset, chosen_transform)
            # initialize model and trainer
            trainer = Train(
                config=(task, dataset, metric, nn_code),
                out_shape=out_shape,
                minimum_accuracy=minimum_accuracy,
                batch=prm['batch'],
                model_name=f"tmp.{temp_filename}",
                task=task,
                train_dataset=train_set,
                test_dataset=test_set,
                metric=metric,
                num_workers=prm.get('num_workers', 1),
                prm=prm,
                save_to_db=save_to_db,
                is_code=True,
                save_path=save_path)
            epoch = prm['epoch']
            result, duration = trainer.train_n_eval(epoch)
            if save_to_db:
                # if result fits the requirement, save the model to database
                if good(result, minimum_accuracy, duration):
                    print("[ERROR] Saving model to database")  # placeholder
                    name = DB_Write.save_nn(nn_code, task, dataset, metric, epoch, prm, force_name=name)
                    print(f"Model saved to database with accuracy: {result}")
                else:
                    print(f"[WARN] Model accuracy {result} is below minimum threshold {minimum_accuracy}. Not saved.")
        except Exception as e:
            print(f"[ERROR] Error during training in train_new: {e}")
            raise

        return name, result, res['score'] / 100.0
