import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import os
import random
from ab.nn.loader.coco_.Caption import GLOBAL_CAPTION_VOCAB

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

# -------------------- Positional Encoding --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, T, d_model]
        return x + self.pe[:, :x.size(1), :]

# -------------------- Encoder --------------------
class CNNEncoder(nn.Module):
    def __init__(self, out_dim=768, dropout=0.2):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        modules = list(resnet.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, out_dim)
        self.dropout = nn.Dropout(dropout)

        for name, param in self.cnn.named_parameters():
            if "7" in name:  # Last block unfrozen
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, images):
        features = self.cnn(images)
        pooled = self.pool(features).flatten(1)
        out = self.fc(pooled)
        out = self.dropout(out)
        return out.unsqueeze(1)  # [B, 1, out_dim]

# -------------------- Decoder --------------------
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=6, nhead=8, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=2048,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None):
        embedded = self.embedding(tgt)        # [B, T, d_model]
        embedded = self.pos_encoding(embedded)
        memory = memory.expand(-1, embedded.size(1), -1)  # [B, T, d_model]
        out = self.transformer_decoder(embedded, memory, tgt_mask=tgt_mask)
        out = self.norm(out)
        return self.fc_out(out)               # [B, T, vocab_size]

# -------------------- Label Smoothing Loss --------------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, ignore_index=0):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        n_class = pred.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_class - 1))
            ignore = target == self.ignore_index
            target = target.masked_fill(ignore, 0)
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            true_dist.masked_fill_(ignore.unsqueeze(1), 0)
        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=1), dim=1))

# -------------------- Main Model Class --------------------
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.vocab_size = out_shape[0]
        self.d_model = 768         # Deep/wide defaults (change here if needed)
        self.n_layers = 6
        self.dropout = prm.get('dropout', 0.2)
        self.encoder = CNNEncoder(out_dim=self.d_model, dropout=self.dropout)
        self.decoder = TransformerDecoder(
            self.vocab_size, d_model=self.d_model, num_layers=self.n_layers, nhead=8, dropout=self.dropout)
        self.word2idx = GLOBAL_CAPTION_VOCAB.get('word2idx', None)
        self.idx2word = GLOBAL_CAPTION_VOCAB.get('idx2word', None)
        self.model_name = "CNNTransformer"
        self.ss_increase = 0.05  # Scheduled sampling rate increase per epoch
        self.ss_prob = 0.0       # Initial scheduled sampling prob

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (LabelSmoothingCrossEntropy(smoothing=0.1, ignore_index=self.word2idx['<PAD>']).to(self.device),)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=3, factor=0.5)
        # Inject vocab from dataloader if present
        train_loader = prm.get('train_loader', None)
        dataset = getattr(train_loader, 'dataset', None) if train_loader is not None else None
        if dataset is not None and hasattr(dataset, 'word2idx'):
            self.word2idx = dataset.word2idx
            self.idx2word = dataset.idx2word

    def scheduled_sampling_prob(self, epoch):
        return min(self.ss_increase * epoch, 0.25)

    def learn(self, train_data, epoch=1):
        self.train()
        self.ss_prob = self.scheduled_sampling_prob(epoch)
        for images, captions in train_data:
            print(f"[DEBUG] BATCH TYPE: images={type(images)}, captions={type(captions)}")
            if isinstance(captions, torch.Tensor):
                print(f"[DEBUG] CAPTIONS SHAPE: {captions.shape}")
            else:
                print(f"[DEBUG] CAPTIONS SAMPLE: {captions[:2]}")
            assert isinstance(captions, torch.Tensor), f"captions is not a tensor: {type(captions)}"

            images = images.to(self.device)
            if captions.ndim > 2:
                captions = captions[:, 0, :].to(self.device)      # [B, T]
            print(f"[MODEL] captions shape: {captions.shape}, dtype: {captions.dtype}, type: {type(captions)}")
            B, T = captions.size()
            tgt_input = captions[:, :-1]                      # [B, T-1]
            tgt_output = captions[:, 1:]                      # [B, T-1]
            self.optimizer.zero_grad()
            memory = self.encoder(images)                     # [B, 1, d_model]
            inputs = tgt_input.clone()
            if self.ss_prob > 0:
                for t in range(1, T-1):
                    use_model = torch.rand(B).to(self.device) < self.ss_prob
                    if use_model.sum() == 0: continue
                    prev_tokens = inputs[use_model, :t]
                    out = self.decoder(prev_tokens, memory[use_model], tgt_mask=None)
                    next_token = out[:, -1, :].argmax(-1)
                    inputs[use_model, t] = next_token
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(inputs.size(1)).to(self.device)
            output = self.decoder(inputs, memory, tgt_mask=tgt_mask)
            loss = self.criteria[0](output.view(-1, self.vocab_size), tgt_output.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def forward(self, images, captions=None, hidden_state=None, max_len=30, beam_size=5, save_path=None, epoch=None, running_save=None):
        self.eval()
        results = []
        batch_size = images.size(0)
        memory = self.encoder(images)    # [B, 1, d_model]
        if captions is not None:
            tgt_input = captions[:, :-1]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)
            return self.decoder(tgt_input, memory, tgt_mask=tgt_mask)
        else:
            # INSTEAD of returning list of strings, return list of token id sequences (as tensor)
            output_seqs = []
            for b in range(batch_size):
                mem = memory[b:b+1, :, :]
                seqs = torch.full((beam_size, 1), self.word2idx['<SOS>'], dtype=torch.long, device=self.device)
                scores = torch.zeros(beam_size, device=self.device)
                finished = [False] * beam_size
                for t in range(max_len):
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(seqs.size(1)).to(self.device)
                    out = self.decoder(seqs, mem.expand(beam_size, -1, -1), tgt_mask=tgt_mask)
                    logprobs = F.log_softmax(out[:, -1, :], dim=-1)
                    topk_logprobs, topk_indices = logprobs.topk(beam_size, dim=-1)
                    if t == 0:
                        seqs = seqs.expand(beam_size, 1)
                        scores = topk_logprobs[0]
                        seqs = torch.cat([seqs, topk_indices[0].unsqueeze(1)], dim=1)
                    else:
                        all_scores = scores.unsqueeze(1) + topk_logprobs
                        all_scores = all_scores.view(-1)
                        best_scores, best_indices = all_scores.topk(beam_size)
                        rows = best_indices // beam_size
                        cols = best_indices % beam_size
                        seqs = torch.cat([seqs[rows], topk_indices[rows, cols].unsqueeze(1)], dim=1)
                        scores = best_scores
                    for k in range(beam_size):
                        if seqs[k, -1].item() == self.word2idx['<EOS>']:
                            finished[k] = True
                    if all(finished):
                        break
                # Pick the best sequence (usually the first beam)
                best = seqs[0].tolist()
                if best[0] == self.word2idx['<SOS>']:
                    best = best[1:]
                if self.word2idx['<EOS>'] in best:
                    best = best[:best.index(self.word2idx['<EOS>'])]
                # Append as tensor of ints (not string!)
                output_seqs.append(torch.tensor(best, dtype=torch.long, device=self.device))
            # Pad all to max_len
            max_out_len = max(len(seq) for seq in output_seqs)
            output_tensor = torch.zeros(batch_size, max_out_len, dtype=torch.long, device=self.device)
            for i, seq in enumerate(output_seqs):
                output_tensor[i, :len(seq)] = seq
            return output_tensor
