import torch
import torch.nn as nn
import torchvision

def supported_hyperparameters():
    # Supported hyperparameters for Optuna or manual tuning
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.hidden_size = 512
        self.vocab_size = out_shape[0]
        self.cnn = ResNetEncoder(self.hidden_size)
        self.rnn = LSTMDecoder(self.vocab_size, hidden_size=self.hidden_size, num_layers=2, dropout=0.3)

    def forward(self, images, captions=None, hidden_state=None):
        features = self.cnn(images)  # (B, 1, 512)
        batch_size = features.size(0)
        hidden = features.permute(1, 0, 2)  # (1, B, 512)
        hidden = hidden.repeat(2, 1, 1)     # (num_layers=2, B, 512)
        cell = torch.zeros(2, batch_size, self.hidden_size, device=self.device)
        hidden_state = (hidden, cell)

        if captions is not None:
            # Training with teacher forcing
            sos_idx = {v: k for k, v in self.__class__.idx2word.items()}.get('<SOS>', 1) if self.__class__.idx2word else 1
            sos_token = torch.full((captions.size(0), 1), sos_idx, dtype=torch.long, device=self.device)
            inputs = torch.cat([sos_token, captions[:, :-1]], dim=1)
            targets = captions
            outputs, _ = self.rnn(features, inputs, hidden_state)
            return outputs, targets
        else:
            # Greedy decoding for inference
            max_len = 20
            sos_idx = {v: k for k, v in self.__class__.idx2word.items()}.get('<SOS>', 1) if self.__class__.idx2word else 1
            inputs = torch.full((batch_size,), sos_idx, dtype=torch.long, device=self.device)
            captions = []
            for _ in range(max_len):
                input_embedded = self.rnn.embedding(inputs).unsqueeze(1)
                output, hidden_state = self.rnn.lstm(input_embedded, hidden_state)
                logits = self.rnn.fc(output.squeeze(1))
                predicted = logits.argmax(1)
                captions.append(predicted.unsqueeze(1))
                inputs = predicted
            return torch.cat(captions, dim=1)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=prm['lr'])

    def learn(self, train_data):
        for i, (images, captions) in enumerate(train_data):
            images = images.to(self.device)
            captions = captions.to(self.device)
            B, C, H, W = images.shape
            N_CAP = captions.shape[1]
            images_exp = images.repeat_interleave(N_CAP, dim=0)
            captions_exp = captions.reshape(-1, captions.shape[-1])

            self.optimizer.zero_grad()
            outputs, targets = self.forward(images_exp, captions_exp)
            loss = self.criteria[0](
                outputs.contiguous().view(-1, outputs.shape[2]),
                targets.contiguous().view(-1)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

            # Print loss for every 300 batches
            if i % 300 == 0:
                print(f"Batch {i}: Loss: {loss.item():.4f}")

    def eval_mode_generate_captions(self, images):
        return self.forward(images)

class ResNetEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        modules = list(backbone.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, output_dim)
        # Freeze CNN backbone
        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.cnn(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x.unsqueeze(1)  # (B, 1, output_dim)

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, hidden_state):
        embedded = self.embedding(captions)
        x, hidden_state = self.lstm(embedded, hidden_state)
        return self.fc(x), hidden_state