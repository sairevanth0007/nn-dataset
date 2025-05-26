import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}  # ✅ removed 'hidden_size'

class Net(nn.Module):
    idx2word = None
    eos_index = None

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.hidden_size = 512  # ✅ hardcoded
        self.vocab_size = out_shape[0]
        self.cnn = ResNetEncoder(in_shape)
        self.rnn = LSTMDecoder(self.vocab_size, hidden_size=self.hidden_size, num_layers=2)

    def forward(self, images, captions=None, hidden_state=None):
        features = self.cnn(images).squeeze(1)
        batch_size = features.size(0)

        hidden = features.unsqueeze(0).repeat(2, 1, 1)
        cell = torch.zeros(2, batch_size, self.hidden_size, device=self.device)
        hidden_state = (hidden, cell)

        if captions is not None:
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            outputs, _ = self.rnn(features, inputs, hidden_state)
            return outputs, targets
        else:
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
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)
            captions = captions[:, 0, :]  # First caption only

            self.optimizer.zero_grad()
            outputs, targets = self.forward(images, captions)
            loss = self.criteria[0](
                outputs.contiguous().view(-1, outputs.shape[2]),
                targets.contiguous().view(-1)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

        # After training loop, print predictions
        if self.__class__.idx2word is not None:
            print("\nSample generated captions (first 5):", flush=True)
            with torch.no_grad():
                generated = self.forward(images)
                if isinstance(generated, tuple):
                    generated = generated[0]
                for i in range(min(5, generated.size(0))):
                    pred_ids = generated[i].tolist()
                    gt_ids = captions[i][1:].tolist()
                    pred_caption = ' '.join(self.__class__.idx2word.get(w, '?') for w in pred_ids if w != 0 and w != self.__class__.eos_index)
                    gt_caption = ' '.join(self.__class__.idx2word.get(w, '?') for w in gt_ids if w != 0 and w != self.__class__.eos_index)
                    print(f"Predicted: {pred_caption}")
                    print(f"GT      : {gt_caption}")


    def eval_mode_generate_captions(self, images):
        return self.forward(images)

class ResNetEncoder(nn.Module):
    def __init__(self, in_shape: tuple):
        super().__init__()
        self.resnet = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, 512)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        return self.fc(x).unsqueeze(1)

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 512)
        self.lstm = nn.LSTM(512, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, hidden_state):
        embedded = self.embedding(captions)
        x, hidden_state = self.lstm(embedded, hidden_state)
        return self.fc(x), hidden_state

    def init_zero_hidden(self, batch: int, device: torch.device):
        return (
            torch.zeros(2, batch, 512, device=device),
            torch.zeros(2, batch, 512, device=device)
        )
