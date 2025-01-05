import ab.nn.util.stat.DB as DB
from pandas import DataFrame

def data(only_best_accuracy=False, task=None, dataset=None, metric=None, nn=None, epoch=None) -> DataFrame:
    """
    Get the NN model code and all related statistics as a pandas dataframe
    for args see :ref:`ab.nn.util.stat.DB.data(...)`
    """
    dt: tuple[dict[str, any]] = DB.data(only_best_accuracy, task=task, dataset=dataset, metric=metric, nn=nn, epoch=epoch)
    return DataFrame.from_records(dt)
