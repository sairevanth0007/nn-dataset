import ab.nn.util.db.Read as DB_Read
import ab.nn.util.Train as Train
from pandas import DataFrame

def data(only_best_accuracy=False, task=None, dataset=None, metric=None, nn=None, epoch=None) -> DataFrame:
    """
    Get the NN model code and all related statistics as a pandas dataframe
    for argument description see :ref:`ab.nn.util.db.Read.data()`
    """
    dt: tuple[dict,...] = DB_Read.data(only_best_accuracy, task=task, dataset=dataset, metric=metric, nn=nn, epoch=epoch)
    return DataFrame.from_records(dt)

def check_nn(nn_code : str, task : str, dataset : str, metric : str, prm: dict) -> str:
    """
    Train the new NN model with the provided hyperparameters (prm) and save it to the database if training is successful.
    for argument description see :ref:`ab.nn.util.db.Write.save_nn()`
    :return: Automatically generated name of NN model.
    """
    return Train.train_new(nn_code, task, dataset, metric, prm)
