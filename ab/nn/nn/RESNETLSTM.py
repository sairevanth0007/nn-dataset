import torch
import torch.nn as nn
import torchvision
import os

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    idx2word = None
    eos_index = None

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.hidden_size = 512
        self.vocab_size = out_shape[0]
        # Use a pretrained ResNet-50 as encoder
        self.cnn = ResNetEncoder(self.hidden_size)
        self.rnn = LSTMDecoder(self.vocab_size, hidden_size=self.hidden_size, num_layers=2, dropout=0.3)

    def forward(self, images, captions=None, hidden_state=None):
        features = self.cnn(images)  # (B, 1, 512)
        batch_size = features.size(0)
        hidden = features.permute(1, 0, 2)  # (1, B, 512)
        hidden = hidden.repeat(2, 1, 1)  # (num_layers=2, B, 512)
        cell = torch.zeros(2, batch_size, self.hidden_size, device=self.device)
        hidden_state = (hidden, cell)

        if captions is not None:
            sos_idx = {v: k for k, v in self.__class__.idx2word.items()}.get('<SOS>', 1) if self.__class__.idx2word else 1
            sos_token = torch.full((captions.size(0), 1), sos_idx, dtype=torch.long, device=self.device)
            inputs = torch.cat([sos_token, captions[:, :-1]], dim=1)
            targets = captions
            outputs, _ = self.rnn(features, inputs, hidden_state)
            return outputs, targets
        else:
            # Greedy decode
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

        last_images, last_captions = None, None
        all_pred_gt = []

        for i, (images, captions) in enumerate(train_data):
            images = images.to(self.device)
            captions = captions.to(self.device)

            # Use all 5 captions per image, flatten for batch.
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

            if i % 300 == 0:
                print(f"Batch {i}: Loss: {loss.item():.4f}")

            last_images = images[:5]
            last_captions = captions[:5]

        # Print & save sample predictions for the last batch
        if self.__class__.idx2word is not None and last_images is not None:
            print("\nSample generated captions (first 5):", flush=True)
            with torch.no_grad():
                generated = self.forward(last_images)
                if isinstance(generated, tuple):
                    generated = generated[0]
                for j in range(min(5, generated.size(0))):
                    pred_ids = generated[j]
                    if hasattr(pred_ids, "tolist"):
                        pred_ids = pred_ids.tolist()
                    if isinstance(pred_ids[0], list):
                        pred_ids = pred_ids[0]  # flatten if needed

                    # Decode predicted sentence (skip special tokens)
                    pred_caption = ' '.join(
                        self.__class__.idx2word.get(w, '?')
                        for w in pred_ids
                        if isinstance(w, int)
                        and w != 0
                        and w != self.__class__.eos_index
                        and w != self.__class__.idx2word.get('<PAD>')
                    )

                    # Decode GT sentence (skip <SOS>, <EOS>, <PAD>)
                    if last_captions is not None:
                        gt_ids = last_captions[j][0].tolist() if last_captions[j].dim() == 2 else last_captions[j].tolist()
                        if isinstance(gt_ids[0], list):
                            gt_ids = gt_ids[0]
                        gt_caption = ' '.join(
                            self.__class__.idx2word.get(w, '?')
                            for w in gt_ids[1:]  # skip <SOS>
                            if isinstance(w, int)
                            and w != 0
                            and w != self.__class__.eos_index
                            and w != self.__class__.idx2word.get('<PAD>')
                        )
                    else:
                        gt_caption = ""

                    print(f"Predicted: {pred_caption}")
                    print(f"GT      : {gt_caption}")
                    all_pred_gt.append((pred_caption, gt_caption))

        # Save predictions to file in the same dir as this file
        save_dir = os.path.dirname(__file__)
        output_file = os.path.join(save_dir, 'coco_predictions.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            for pred, gt in all_pred_gt:
                f.write(f"Predicted: {pred}\nGT      : {gt}\n\n")
        print(f"Saved predictions to {output_file}")

    def eval_mode_generate_captions(self, images):
        return self.forward(images)

class ResNetEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        modules = list(backbone.children())[:-2]  # Remove avgpool & fc
        self.cnn = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, output_dim)
        # Optionally, freeze CNN backbone for faster training:
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
