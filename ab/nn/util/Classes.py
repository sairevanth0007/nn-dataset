import time as time

from tqdm import tqdm

import ab.nn.util.Const as Const
from ab.nn.util.Exception import *


class DataRoll(tqdm):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.it = super().__iter__()
        self.init_time = time.time()

    def __iter__(self):
        return self

    def __next__(self):
        if self.n > 5:
            duration = max(1e-1, time.time() - self.init_time)
            estimated_time = self.total * duration  / self.n
            if estimated_time > Const.max_epoch_seconds:
                raise LearnTimeException(estimated_time, Const.max_epoch_seconds, duration)
        return self.it.__next__()
