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

    def __init__(self, input_size: int, hidden_size: int, output_size: int, batch: int, num_layers: int = 1) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch = batch
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state) -> tuple[torch.Tensor, tuple]:
        x, hidden_state = self.lstm(x, hidden_state)
        out = self.h2o(x)
        return out, hidden_state

    def init_zero_hidden(self, batch=None) -> tuple[torch.Tensor, torch.Tensor]:
        if batch is None:
            batch = self.batch
        h_0 = torch.zeros(self.num_layers, batch, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, batch, self.hidden_size)
        return (h_0, c_0)
