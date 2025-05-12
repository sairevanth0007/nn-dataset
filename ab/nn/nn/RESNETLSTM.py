import torch
import torch.nn as nn

def supported_hyperparameters():
    # Added hidden_size as a supported hyperparameter.
    return {'lr', 'momentum', 'hidden_size'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.cnn = ResNetEncoder(in_shape)
        # LSTMDecoder no longer embeds tokens itself
        self.rnn = LSTMDecoder(out_shape[0], hidden_size=512, num_layers=2)
    
    # now receives pre-embedded captions
    def forward(self, images, caption_emb, hidden_state):
        features = self.cnn(images)
        outputs, hidden_state = self.rnn(features, caption_emb, hidden_state)
        return outputs, hidden_state
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])
    
    # now unpacks (image, embedded_caption, target_ids)
    def learn(self, train_data):
        for images, caption_emb, target_ids in train_data:
            images       = images.to(self.device)
            caption_emb  = caption_emb.to(self.device)   # [B, T, D]
            targets      = target_ids.to(self.device)    # [B, T]

            hidden_state = self.rnn.init_zero_hidden(batch=images.size(0),
                                                     device=self.device)
            hidden_state = tuple(h.to(self.device) for h in hidden_state)
            self.optimizer.zero_grad()
            outputs, _ = self(images, caption_emb, hidden_state)
            # outputs: [B, T, V] → flatten for CrossEntropyLoss
            loss = self.criteria[0](
                outputs.contiguous().view(-1, outputs.shape[2]),
                targets.contiguous().view(-1)
            )

            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()




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
        x = self.fc(x)
        return x.unsqueeze(1)  # Add time dimension for LSTM

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        # embedding removed—data loader provides caption_emb
        self.lstm = nn.LSTM(512, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    # now expects caption_emb directly
    def forward(self, features, caption_emb, hidden_state):
        x, hidden_state = self.lstm(caption_emb, hidden_state)
        out = self.fc(x)
        return out, hidden_state
    
    def init_zero_hidden(self, batch: int, device: torch.device):
        return (torch.zeros(2, batch, 512, device=device),
                torch.zeros(2, batch, 512, device=device))
