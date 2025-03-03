import torch
from ab.nn.metric.base.base import BaseMetric

class Accuracy(BaseMetric):
    def reset(self):
        self.correct = 0
        self.total = 0
    
    def update(self, outputs, targets):
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        self.correct += correct
        self.total += total
    
    def __call__(self, outputs, targets):
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        self.update(outputs, targets)
        return correct, total
    
    def result(self):
        return self.correct / max(self.total, 1e-8)

# Function to create metric instance
def create_metric(out_shape=None):
    return Accuracy()
