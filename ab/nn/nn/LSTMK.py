import torch
import torch.nn as nn

def supported_hyperparameters():
    # Added hidden_size as a supported hyperparameter.
    return {'lr', 'momentum', 'hidden_size'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, num_layers=1):
        super().__init__()
        # Assume in_shape is a tuple (e.g. (batch, seq_length, feature_dim))
        if isinstance(in_shape, (list, tuple)):
            # Use the first element as batch size and the last element as input size
            self.batch = in_shape[0] if len(in_shape) > 0 else 1
            self.input_size = in_shape[-1]
        else:
            self.batch = 1
            self.input_size = in_shape

        # Get hidden_size from hyperparameters, defaulting to 256 if not provided.
        #self.hidden_size = prm.get('hidden_size', 256)
        self.hidden_size = 256
        
        
        # For out_shape, if it’s a tuple, take the first element as the output size.
        if isinstance(out_shape, (list, tuple)):
            self.output_size = out_shape[0] if len(out_shape) > 0 else out_shape
        else:
            self.output_size = out_shape

        # num_layers defaults to 1 if not provided.
        self.num_layers = num_layers if num_layers is not None else 1
        self.device = device

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.h2o = nn.Linear(self.hidden_size, self.output_size)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    

    def learn(self, train_data):
        for images, captions in train_data:
            images, captions = images.to(self.device), captions.to(self.device)
            # Use only the first caption from the captions tensor.
            targets = captions[:, 0, :]  # shape: [batch, target_len]
            self.optimizer.zero_grad()
            outputs = self(images)  # outputs has shape [batch, self.max_len, vocab_size]
            # Slice outputs to match target caption length.
            target_len = targets.size(1)
            outputs = outputs[:, :target_len, :]
            # Reshape outputs and targets for CrossEntropyLoss:
            loss = self.criteria[0](outputs.reshape(-1, self.output_size), targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()



    def forward(self, x, hidden_state=None):
        if x.dim() == 4:
            # Instead of flattening all dimensions (which yields 268203 features for a 3x299x299 image),
            # we average over the channel and height dimensions to produce a tensor of shape (batch, 299).
            x = x.mean(dim=1).mean(dim=1)  # Now shape is (batch, 299)
            x = x.unsqueeze(1)            # Shape becomes (batch, 1, 299)
        # Automatically initialize hidden state if none is provided.
        if hidden_state is None:
            hidden_state = self.init_zero_hidden(batch=x.size(0))
        x, hidden_state = self.lstm(x, hidden_state)
        out = self.h2o(x)
        return out

    def init_zero_hidden(self, batch=None):
        if batch is None:
            batch = self.batch
        # Create hidden states on the correct device.
        h_0 = torch.zeros(self.num_layers, batch, self.hidden_size, device=self.device)
        c_0 = torch.zeros(self.num_layers, batch, self.hidden_size, device=self.device)
        return (h_0, c_0)
