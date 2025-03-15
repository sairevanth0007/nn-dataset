import ab.nn.util.db.Read as DB_Read
import ab.nn.util.Train as Train
import ab.nn.util.Util as Util
from pandas import DataFrame

def data(only_best_accuracy=False, task=None, dataset=None, metric=None, nn=None, epoch=None, cast_prm=True) -> DataFrame:
    """
    Get the NN model code and all related statistics as a pandas DataFrame.

    For the detailed description of arguments see :ref:`ab.nn.util.db.Read.data()`.
    
    Parameters:
      - only_best_accuracy (bool): If True, for each unique combination of 
          (task, dataset, metric, nn, epoch) only the row with the highest accuracy is returned.
          If False, all matching rows are returned.
      - task, dataset, metric, nn, epoch: Optional filters to restrict the results.

    Returns:
      - A pandas DataFrame where each row is a dictionary containing:
          'task', 'dataset', 'metric', 'metric_code',
          'nn', 'nn_code', 'epoch', 'accuracy', 'duration',
          'prm', and 'transform_code'.
    """
    dt: tuple[dict, ...] = DB_Read.data(only_best_accuracy, task=task, dataset=dataset, metric=metric, nn=nn, epoch=epoch, cast_prm=cast_prm)
    return DataFrame.from_records(dt)

def check_nn(nn_code : str, task : str, dataset : str, metric : str, prm: dict, save_to_db=True, prefix = None, save_path = None) -> tuple[str, float, int]:
    """
    Train the new NN model with the provided hyperparameters (prm) and save it to the database if training is successful.
    for argument description see :ref:`ab.nn.util.db.Write.save_nn()`
    :return: Automatically generated name of NN model and its accuracy.
    """
    return Train.train_new(nn_code, task, dataset, metric, prm, save_to_db=save_to_db, prefix = prefix, save_path = save_path)


def accuracy_to_time_metric(accuracy, min_accuracy, training_duration) -> float:
    """
        Accuracy to time metric (for fixed number of training epochs) is essential for detecting the fastest accuracy improvements during neural network training.
        """
    return Util.accuracy_to_time_metric(accuracy, min_accuracy, training_duration)