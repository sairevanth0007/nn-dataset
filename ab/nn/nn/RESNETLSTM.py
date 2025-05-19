import torch
import torch.nn as nn
import torchvision

def supported_hyperparameters():
    return {'lr', 'dropout'}

class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512)
        )
        
    def forward(self, x):
        features = self.cnn(x)
        return self.projection(features)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size=512, hidden_size=512, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def forward(self, features, captions=None):
        batch_size = features.size(0)
        h = torch.zeros(batch_size, self.hidden_size, device=features.device)
        c = torch.zeros(batch_size, self.hidden_size, device=features.device)
        
        if captions is not None:
            # Handle both single and multiple captions
            if captions.dim() == 3:
                captions = captions[:, 0, :]  # Use first caption
            
            # Ensure we predict for sequence length = input length - 1
            seq_len = captions.size(1)
            outputs = torch.zeros(batch_size, seq_len, self.fc.out_features, 
                               device=features.device)
            
            for t in range(seq_len):
                embeddings = self.dropout(self.embed(captions[:, t]))
                lstm_input = torch.cat([embeddings, features], dim=1)
                h, c = self.lstm(lstm_input, (h, c))
                outputs[:, t] = self.fc(h)
            return outputs
        else:
            return self.generate(features, h, c)

    def generate(self, features, h, c, max_len=20):
        outputs = []
        # using ones() which infers the shape correctly
        inputs = torch.ones(features.size(0), dtype=torch.long, device=features.device)

        for _ in range(max_len):
            embeddings = self.dropout(self.embed(inputs))
            lstm_input = torch.cat([embeddings, features], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            outputs.append(self.fc(h).argmax(1))
            inputs = outputs[-1]
        return torch.stack(outputs, dim=1)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.encoder = VisualEncoder()
        self.decoder = Decoder(out_shape[0], dropout=float(prm.get('dropout', 0.3)))
        self.device = device
        self.to(device)

    def forward(self, images, captions=None):
        features = self.encoder(images.to(self.device))
        return self.decoder(features, captions.to(self.device) if captions is not None else None)

    def train_setup(self, prm):
        self.optimizer = torch.optim.AdamW(self.parameters(),
                                  lr=prm['lr'], weight_decay=1e-5)
        # Define n_total_steps, e.g., as a parameter or a fixed value
        n_total_steps = prm.get('n_total_steps', 100)  # Set default or pass in prm
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_total_steps, eta_min=prm['lr'] * 0.01
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def learn(self, train_loader):
        self.train()
        total_loss = 0
        for images, captions in train_loader:
            images = images.to(self.device)
            captions = captions.to(self.device)
            
            # Ensure consistent dimensions
            if captions.dim() == 3:
                captions = captions[:, 0, :]  # Use first caption
            
            # Input: all tokens except last, Target: all tokens except first
            outputs = self(images, captions[:, :-1])  # Predict next tokens
            targets = captions[:, 1:]  # Shifted by one
            
            # Final verification
            assert outputs.size(0) == targets.size(0), "Batch size mismatch"
            assert outputs.size(1) == targets.size(1), f"Seq len mismatch: {outputs.size(1)} vs {targets.size(1)}"
            
            loss = self.criterion(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
