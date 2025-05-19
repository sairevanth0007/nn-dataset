import os
import requests
from collections import Counter
from os import makedirs
from os.path import join, exists

import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_and_extract_archive
from torchvision import transforms

from ab.nn.util.Const import data_dir

# COCO URLs
default_coco_ann = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
default_coco_img = 'http://images.cocodataset.org/zips/{}2017.zip'

# Normalization stats
__norm_mean = [0.485, 0.456, 0.406]
__norm_std  = [0.229, 0.224, 0.225]

minimum_bleu = 0.04

class COCOCaptionDataset(Dataset):
    def __init__(self, root, split='train', transform=None, max_length=20,
                 ann_url=default_coco_ann, img_url=default_coco_img):
        if split not in ('train', 'val'):
            raise ValueError(f"Invalid split: {split}")
        self.root = root
        self.split = split
        self.transform = transform
        self.max_length = max_length

        # Download annotations if missing
        ann_dir = join(root, 'annotations')
        if not exists(ann_dir):
            makedirs(root, exist_ok=True)
            download_and_extract_archive(ann_url, root,
                                         filename='annotations_trainval2017.zip')
        ann_file = join(ann_dir, f'captions_{split}2017.json')
        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.imgs.keys())

        # Download images if missing
        img_dir = join(root, f'{split}2017')
        first = self.coco.loadImgs(self.ids[0])[0]
        if not exists(join(img_dir, first['file_name'])):
            download_and_extract_archive(img_url.format(split), root,
                                         filename=f'{split}2017.zip')
        self.img_dir = img_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        path = join(self.img_dir, info['file_name'])

        # Load (or download) image
        try:
            with Image.open(path) as img:
                image = img.convert('RGB')
        except Exception:
            resp = requests.get(info['coco_url'])
            resp.raise_for_status()
            makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(resp.content)
            with Image.open(path) as img:
                image = img.convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Load captions (take first for simplicity)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        captions = [a['caption'] for a in anns][:1]

        return image, captions

    def collate_fn(self, batch):
        # Use dataset-specific vocab mapping
        w2i = getattr(self, 'word2idx', None)
        if w2i is None:
            raise ValueError("Dataset missing 'word2idx' for collate_fn")

        images, cap_lists = zip(*batch)
        images = torch.stack(images, dim=0)

        sequences = []
        for caps in cap_lists:
            seq = [w2i['<SOS>']]
            seq += [w2i.get(w.lower(), w2i['<UNK>']) for w in caps[0].split()]
            seq = seq[:self.max_length]
            seq.append(w2i['<EOS>'])
            sequences.append(seq)

        max_len = max(len(s) for s in sequences)
        padded = [s + [w2i['<PAD>']] * (max_len - len(s)) for s in sequences]
        captions_tensor = torch.tensor(padded, dtype=torch.long)
        return images, captions_tensor


def build_vocab(dataset, threshold=5):
    counter = Counter()
    for _, caps in dataset:
        for c in caps:
            counter.update(c.lower().split())

    specials = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    words = sorted(w for w, cnt in counter.items() if cnt >= threshold)
    vocab = specials + words
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(__norm_mean, __norm_std)
    ])


def loader(transform_fn, task):
    if task not in ('img-captioning', 'image-captioning'):
        raise ValueError(f"Task '{task}' not supported by Caption loader")

    try:
        transform = transform_fn()
    except TypeError:
        transform = get_transform()

    path = join(data_dir, 'coco')
    train_ds = COCOCaptionDataset(path, 'train', transform)
    val_ds   = COCOCaptionDataset(path, 'val', transform)

    vocab_path = join(path, 'vocab.pth')
    if exists(vocab_path):
        vdata = torch.load(vocab_path)
        w2i, i2w = vdata['word2idx'], vdata['idx2word']
    else:
        w2i, i2w = build_vocab(train_ds, threshold=5)
        torch.save({'word2idx': w2i, 'idx2word': i2w}, vocab_path)

    for ds in (train_ds, val_ds):
        ds.word2idx = w2i
        ds.idx2word = i2w

    vocab_size = len(w2i)
    return (vocab_size,), minimum_bleu, train_ds, val_ds
