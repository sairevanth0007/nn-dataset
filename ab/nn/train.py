import optuna
from torch.cuda import OutOfMemoryError

from ab.nn.util.Const import *
from ab.nn.util.Loader import Loader
from ab.nn.util.Train import Train
from ab.nn.util.Util import merge_prm, get_attr, conf_to_names, max_batch, CudaOutOfMemory, ModelException, args
from ab.nn.util.stat.Calc import patterns_to_configs
from ab.nn.util.stat.DB import count_trials_left


def main(config: str | tuple = default_config, n_epochs: int = default_epochs,
         n_optuna_trials: int | str = default_trials,
         min_batch_binary_power: int = default_min_batch_power, max_batch_binary_power: int = default_max_batch_power,
         min_learning_rate: float = default_min_lr, max_learning_rate: float = default_max_lr,
         min_momentum: float = default_min_momentum, max_momentum: float = default_max_momentum,
         transform: str = None, nn_fail_attempts: int = default_nn_fail_attempts, random_config_order:bool = default_random_config_order):
    """
    Main function for training models using Optuna optimization.
    :param config: Configuration specifying the model training pipelines. The default value for all configurations.
    :param n_epochs: Number of training epochs.
    :param n_optuna_trials: The total number of Optuna trials the model should have. If negative, its absolute value represents the number of additional trials.
    :param min_batch_binary_power: Minimum power of two for batch size. E.g., with a value of 0, batch size equals 2**0 = 1.
    :param max_batch_binary_power: Maximum power of two for batch size. E.g., with a value of 12, batch size equals 2**12 = 4096.
    :param min_learning_rate: Minimum value of learning rate.
    :param max_learning_rate: Maximum value of learning rate.
    :param min_momentum: Minimum value of momentum.
    :param max_momentum: Maximum value of momentum.
    :param transform: The transformation algorithm name. If None (default), all available algorithms are used by Optuna.
    :param nn_fail_attempts: Number of attempts if the neural network model throws exceptions.
    :param random_config_order: If random shuffling of the config list is required.
    """

    if min_batch_binary_power > max_batch_binary_power: raise Exception(f"min_batch_binary_power {min_batch_binary_power} > max_batch_binary_power {max_batch_binary_power}")
    if min_learning_rate > max_learning_rate: raise Exception(f"min_learning_rate {min_learning_rate} > max_learning_rate {max_learning_rate}")
    if min_momentum > max_momentum: raise Exception(f"min_momentum {min_momentum} > max_momentum {max_momentum}")

    # Initialize the SQLite database
    # initialize_database() # todo Change to align with the new functionality
    # Determine configurations based on the provided config
    sub_configs = patterns_to_configs(config, random_config_order)

    print(f"Training configurations ({n_epochs} epochs):")
    for idx, sub_config in enumerate(sub_configs, start=1):
        print(f"{idx}. {sub_config}")
    for sub_config in sub_configs:
            task, dataset_name, metric, model_name = conf_to_names(sub_config)
            model_stat_dir: str = join(stat_dir, sub_config)
            trials_file = join(model_stat_dir, str(n_epochs), 'trials.json')
            n_optuna_trials_left = count_trials_left(trials_file, model_name, n_optuna_trials)
            if n_optuna_trials_left == 0:
                print(f"The model {model_name} has already passed all trials for task: {task}, dataset: {dataset_name},"
                      f" metric: {metric}, epochs: {n_epochs}")
            else:
                print(f"\nStarting training for the model: {model_name}, task: {task}, dataset: {dataset_name},"
                      f" metric: {metric}, epochs: {n_epochs}")
                fail_iterations = nn_fail_attempts
                continue_study = True
                max_batch_binary_power_local = max_batch_binary_power
                while continue_study and max_batch_binary_power_local >= min_batch_binary_power and fail_iterations > -1:
                    try:
                        # Configure Optuna for the current model
                        def objective(trial):
                            try:
                                # Load model
                                s_prm: set = get_attr(f"dataset.{model_name}", "supported_hyperparameters")()
                                # Suggest hyperparameters
                                prms = {}
                                for prm in s_prm:
                                    if 'lr' == prm:
                                        prms[prm] = trial.suggest_float('lr', min_learning_rate, max_learning_rate, log=True)
                                    elif 'momentum' == prm:
                                        prms[prm] = trial.suggest_float('momentum', min_momentum, max_momentum, log=False)
                                    else:
                                        prms[prm] = trial.suggest_float(prm, 0.0, 1.0, log=False)
                                batch = trial.suggest_categorical('batch', [max_batch(x) for x in range(min_batch_binary_power, max_batch_binary_power_local + 1)])
                                transform_name = trial.suggest_categorical('transform',
                                                                           [transform] if transform is not None else ['cifar-10_complex_32', 'cifar-10_norm_32', 'cifar-10_norm_299', 'cifar-10_norm_512', 'echo'])
                                prms = merge_prm(prms, {'batch': batch, 'transform': transform_name})
                                prm_str = ''
                                for k, v in prms.items():
                                    prm_str += f", {k}: {v}"
                                print(f"Initialize training with {prm_str[2:]}")
                                # Load dataset
                                try:
                                    out_shape, train_set, test_set = Loader.load_dataset(dataset_name, transform_name)
                                except Exception as e:
                                    print(f"Skipping model '{model_name}': failed to load dataset. Error: {e}")
                                    return 0.0

                                # Initialize model and trainer
                                if task == 'txt-generation':
                                    # Dynamically import RNN or LSTM model
                                    if model_name.lower() == 'rnn':
                                        from ab.nn.dataset.RNN import Net as RNNNet
                                        model = RNNNet(1, 256, len(train_set.chars), batch)
                                    elif model_name.lower() == 'lstm':
                                        from ab.nn.dataset.LSTM import Net as LSTMNet
                                        model = LSTMNet(1, 256, len(train_set.chars), batch, num_layers=2)
                                    else:
                                        raise ValueError(f"Unsupported text generation model: {model_name}")
                                return Train(sub_config, out_shape, batch, model_name, model_stat_dir,
                                                task, train_set, test_set, metric, prms).train_n_eval(n_epochs)
                            except Exception as e:
                                if isinstance(e, OutOfMemoryError):
                                    if max_batch_binary_power_local <= min_batch_binary_power:
                                        return 0.0
                                    else:
                                        raise CudaOutOfMemory(batch)
                                else:
                                    print(f"error '{model_name}': failed to train. Error: {e}")
                                    if fail_iterations < 0:
                                        return 0.0
                                    else:
                                        raise ModelException()
                        # Launch Optuna for the current NN model
                        study = optuna.create_study(study_name=model_name, direction='maximize')
                        study.optimize(objective, n_trials=n_optuna_trials_left)
                        continue_study = False
                    except ModelException:
                        fail_iterations -= 1
                    except CudaOutOfMemory as e:
                        max_batch_binary_power_local = e.batch_size_power() - 1
                        print(f"Max batch is decreased to {max_batch(max_batch_binary_power_local)} due to a CUDA Out of Memory Exception for model '{model_name}'")


if __name__ == "__main__":
    a = args()
    main(a.config, a.epochs, a.trials, a.min_batch_binary_power, a.max_batch_binary_power,
         a.min_learning_rate, a.max_learning_rate, a.min_momentum, a.max_momentum, a.transform,
         a.nn_fail_attempts, a.random_config_order)
