import torch
import torch.nn as nn


def supported_hyperparameters():
    return {'lr', 'momentum'}


class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, input_size: int = 128, hidden_size: int = 256, output_size: int = 10, batch: int = 1, num_layers: int = 1) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch = batch
        self.num_layers = num_layers

        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.i2h(x)
        hidden_state = self.h2h(hidden_state)
        hidden_state = torch.tanh(x + hidden_state)
        out = self.h2o(hidden_state)
        return out, hidden_state

    def init_zero_hidden(self, batch=1) -> torch.Tensor:
        return torch.zeros(batch, self.hidden_size, requires_grad=False)
