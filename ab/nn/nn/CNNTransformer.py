import torch
import torch.nn as nn
import torchvision.models as models
import math
import os
from ab.nn.loader.coco_.Caption import GLOBAL_CAPTION_VOCAB

def supported_hyperparameters():
    return {'lr', 'momentum'}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        modules = list(resnet.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 768)

    def forward(self, images):
        features = self.cnn(images)  # [B, 2048, H, W]
        pooled = self.pool(features).flatten(1)  # [B, 2048]
        return self.fc(pooled).unsqueeze(1)  # [B, 1, 768]


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=6, nhead=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        embedded = self.embedding(tgt)  # [T, B, 512]
        embedded = self.pos_encoding(embedded)
        tgt_mask = nn.Transformer().generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        out = self.transformer_decoder(embedded, memory, tgt_mask=tgt_mask)
        return self.fc_out(out)  # [T, B, vocab_size]


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.encoder = CNNEncoder()
        self.decoder = TransformerDecoder(out_shape[0])
        self.vocab_size = out_shape[0]
        self.word2idx = None
        self.idx2word = None
        self.model_name = "CNNTransformer"

        self.word2idx = GLOBAL_CAPTION_VOCAB.get('word2idx', None)
        self.idx2word = GLOBAL_CAPTION_VOCAB.get('idx2word', None)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=self.word2idx['<PAD>']).to(self.device),)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'])
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.8)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=3, factor=0.5, verbose=True)

        train_loader = prm.get('train_loader', None)
        dataset = getattr(train_loader, 'dataset', None) if train_loader is not None else None
        if dataset is not None and hasattr(dataset, 'word2idx'):
            self.word2idx = dataset.word2idx
            self.idx2word = dataset.idx2word

    def learn(self, train_data):

        if self.word2idx is None or self.idx2word is None:
            if hasattr(train_data, 'dataset'):
                self.word2idx = getattr(train_data.dataset, 'word2idx', None)
                self.idx2word = getattr(train_data.dataset, 'idx2word', None)
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)[:, 0, :]  # [B, T]

            tgt_input = captions[:, :-1].T  # [T-1, B]
            tgt_output = captions[:, 1:].T  # [T-1, B]

            self.optimizer.zero_grad()
            memory = self.encoder(images).transpose(0, 1)  # [1, B, 512]
            output = self.decoder(tgt_input, memory)  # [T-1, B, vocab_size]

            loss = self.criteria[0](output.view(-1, output.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
            #self.scheduler.step()  # Only for StepLR, not for ReduceLROnPlateau

    def forward(self, images, captions=None, hidden_state=None):
        print(f"[DEBUG] Model forward called. Images shape: {images.shape}")
        if self.word2idx is None or self.idx2word is None:
            raise ValueError("word2idx and idx2word must be set before evaluation.")
        memory = self.encoder(images).transpose(0, 1)
        if captions is not None:
            # training
            tgt_input = captions[:, :-1]
            
            return self.decoder(tgt_input, memory).transpose(0, 1)

        batch_size = images.size(0)
        results = []
        max_len = 30
        #print(f"[DEBUG] Starting generation for batch of {batch_size} images")
        for i in range(batch_size):
            #print(f"[DEBUG] Generating caption for image {i+1}/{batch_size}")
            generated = [self.word2idx['<SOS>']]
            mem = memory[:, i:i+1, :]
            for t in range(max_len):
                tgt = torch.tensor(generated, dtype=torch.long).unsqueeze(1).to(self.device)
                out = self.decoder(tgt, mem)
                next_token = out.argmax(dim=-1)[-1].item()
                generated.append(next_token)
                if next_token == self.word2idx['<EOS>']:
                    #print(f"[DEBUG] Image {i+1}: Stopped at step {t+1} with <EOS>")
                    break
            results.append(generated[1:])
        #print("[DEBUG] Finished generation for all images in batch.")

        # Save to file
        os.makedirs("output/generated_captions", exist_ok=True)
        with open("output/generated_captions/{}_captions.txt".format(self.model_name), "w") as f:
            for seq in results:
                caption = ' '.join([self.idx2word.get(idx, '<UNK>') for idx in seq if idx in self.idx2word])
                f.write(caption + "\n")

        # Pad results and create logits tensor
        max_seq_len = max(len(r) for r in results)
        preds = torch.zeros(batch_size, max_seq_len, self.vocab_size).to(self.device)
        for i, seq in enumerate(results):
            for t, idx in enumerate(seq):
                preds[i, t, idx] = 1.0  # one-hot
        return preds

    def generate(self, image, word2idx, idx2word, max_len=20):
        self.eval()
        self.word2idx = word2idx
        self.idx2word = idx2word
        with torch.no_grad():
            image = image.unsqueeze(0).to(self.device)         # [1, C, H, W]
            memory = self.encoder(image).transpose(0, 1)        # [1, 1, 512]

            generated = [word2idx['<SOS>']]
            for _ in range(max_len):
                tgt = torch.tensor(generated, dtype=torch.long).unsqueeze(1).to(self.device)  # [T, 1]
                out = self.decoder(tgt, memory)        # [T, 1, vocab_size]
                next_token = out.argmax(dim=-1)[-1].item()
                generated.append(next_token)
                if next_token == word2idx['<EOS>']:
                    break

            caption = [idx2word[idx] for idx in generated[1:-1]]  # exclude <SOS> and <EOS>
            return ' '.join(caption)