import math

class CudaOutOfMemory(Exception):
    def __init__(self, batch):
        self.batch_power = int(math.log2(batch))

    def batch_size_power(self):
        return self.batch_power


class NNException(Exception):
    def __init__(self):
        pass


class AccuracyException(Exception):
    def __init__(self, accuracy, message):
        self.accuracy = accuracy
        self.message = message


class LearnTimeException(Exception):
    def __init__(self, estimated_training_time, max_learn_seconds):
        self.estimated_training_time = estimated_training_time
        self.max_learn_seconds = max_learn_seconds