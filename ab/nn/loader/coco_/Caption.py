import os
from os import makedirs
from os.path import join, exists
import requests
from collections import Counter

import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_and_extract_archive
from typing import List, Dict, Tuple, Any

from ab.nn.util.Const import data_dir

# COCO URLs
coco_ann_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
coco_img_url = 'http://images.cocodataset.org/zips/{}2017.zip'

# For consistency with your other files:
__norm_mean = (104.01362025, 114.03422265, 119.9165958)
__norm_dev = (73.6027665, 69.89082075, 70.9150767)
minimum_bleu = 0.05

class COCOCaptionDataset(Dataset):
    
    def __init__(self, transform, root, split='train'):
        """
        Parameters
        ----------
        transform : callable
            Image transformation (e.g., normalization, resizing)
        root : str
            Path to the COCO dataset root directory
        split : str
            'train' or 'val'
        """
        super().__init__()
        valid_splits = ['train', 'val']
        if split not in valid_splits:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

        self.root = root
        self.transform = transform
        self.split = split
        
        # Ensure the annotation file is present; download if missing.
        ann_dir = os.path.join(root, 'annotations')
        if not os.path.exists(ann_dir):
            makedirs(root, exist_ok=True)
            download_and_extract_archive(coco_ann_url, root, filename='annotations_trainval2017.zip')

        # Use the 'captions_{split}2017.json' file
        ann_file = os.path.join(ann_dir, f'captions_{split}2017.json')
        if not os.path.exists(ann_file):
            raise RuntimeError(f"Missing {ann_file}. Check that 'annotations_trainval2017.zip' was properly extracted.")

        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        self.img_dir = os.path.join(root, f'{split}2017')
        # Check if images exist; if not, download
        first_image_info = self.coco.loadImgs(self.ids[0])[0]
        first_file_path = os.path.join(self.img_dir, first_image_info['file_name'])
        if not os.path.exists(first_file_path):
            download_and_extract_archive(coco_img_url.format(split), root, filename=f'{split}2017.zip')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        Returns (image, captions) for a given index.
        By default, all captions for that image are returned.
        """
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        file_path = os.path.join(self.img_dir, img_info['file_name'])

        try:
            with Image.open(file_path) as img_file:
                image = img_file.convert('RGB')
                image = image.copy()
        except:
            if not hasattr(self, 'no_missing_img'):
                self.no_missing_img = True
            response = requests.get(img_info['coco_url'])
            if response.status_code == 200:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                with Image.open(file_path) as img_file:
                    image = img_file.convert('RGB')
                    image = image.copy()
            else:
                raise RuntimeError(f"Failed to download image: {img_info['file_name']}")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        captions = [ann['caption'] for ann in anns]

        if self.transform is not None:
            image = self.transform(image)
        image = image.squeeze(0) if image.dim() == 4 and image.size(0) == 1 else image
        return image, captions 
         
    @staticmethod
    def collate_fn(batch, word2idx):
        """
        Collate function for DataLoader.
        Converts captions to tensors and pads them to the same length, and also pads the number of captions per image.
        """
        images = []
        all_captions = []

        for (img, caps) in batch:
            images.append(img)
            tokenized_captions = [
                [word2idx.get(word, word2idx['<UNK>']) for word in cap.lower().split()] for cap in caps
            ]
            all_captions.append(tokenized_captions)

        images = torch.stack(images, dim=0)

        max_len = max(len(cap) for caps in all_captions for cap in caps)
        max_captions = max(len(caps) for caps in all_captions)

        padded_captions = []
        for caps in all_captions:
            padded_caps = [cap + [word2idx['<PAD>']] * (max_len - len(cap)) for cap in caps]
            num_to_pad = max_captions - len(caps)
            for _ in range(num_to_pad):
                padded_caps.append([word2idx['<PAD>']] * max_len)
            padded_captions.append(torch.tensor(padded_caps))

        captions_tensor = torch.stack(padded_captions, dim=0)
        return images, captions_tensor

def build_vocab(dataset, threshold=1):
    """
    Build a vocabulary from the captions in the given dataset.
    
    Parameters
    ----------
    dataset : Dataset
        The dataset from which to build the vocabulary.
    threshold : int
        Minimum frequency a word must have to be included.
    
    Returns
    -------
    word2idx : dict
        Mapping from word to index.
    idx2word : dict
        Mapping from index to word.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    counter = Counter()
    for i in range(len(dataset)):
        _, captions = dataset[i]
        for caption in captions:
            tokens = caption.lower().split()
            counter.update(tokens)

    specials = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    vocab_words = sorted([word for word, count in counter.items() if count >= threshold])
    vocab = specials + vocab_words
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def loader(transform_fn, task):
    if task != 'image-captioning':
        raise Exception(f"The task '{task}' is not implemented in this file.")
     
    transform = transform_fn((__norm_mean, __norm_dev))
    path = join(data_dir, 'coco')
    
    train_dataset = COCOCaptionDataset(transform=transform, root=path, split='train')
    val_dataset = COCOCaptionDataset(transform=transform, root=path, split='val')

    vocab_path = os.path.join(path, 'vocab.pth')
    if os.path.exists(vocab_path):
        vocab_data = torch.load(vocab_path)
        word2idx = vocab_data['word2idx']
        idx2word = vocab_data['idx2word']
    else:
        word2idx, idx2word = build_vocab(train_dataset, threshold=1)
        torch.save({'word2idx': word2idx, 'idx2word': idx2word}, vocab_path)

    train_dataset.word2idx = word2idx
    train_dataset.idx2word = idx2word
    val_dataset.word2idx = word2idx
    val_dataset.idx2word = idx2word

    vocab_size = len(word2idx)
    return (vocab_size,), minimum_bleu, train_dataset, val_dataset
