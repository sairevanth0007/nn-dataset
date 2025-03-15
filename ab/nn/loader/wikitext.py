import torch
from torch.utils.data import Dataset
from datasets import load_dataset


minimum_accuracy = 0.01 # todo: Required correct value for minimum accuracy provided by the untrained classifier due to random output generation.

def loader(transform_fn, task):
    dataset_name = "Salesforce/wikitext"
    config = "wikitext-2-raw-v1"
    seq_length = 100
    dataset = load_dataset(dataset_name, config)
    data = "\n".join(dataset["train"]["text"]).lower()
    txt_dataset = TextDatasetPreparation(data, seq_length)
    return None, minimum_accuracy, txt_dataset, txt_dataset

class TextDatasetPreparation(Dataset):
    """
    A dataset class for preparing character-level text data for tasks like text generation.
    :param txt_data: Input text as a single string.
    :param seq_length: Length of each input sequence (default: 25).
    :return: Pairs of input and target sequences as tensors.
    """
    def __init__(self, txt_data: str, seq_length: int = 25):
        self.chars = sorted(list(set(txt_data)))
        self.data_size, self.vocab_size = len(txt_data), len(self.chars)
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.seq_length = seq_length
        self.X = self.string_to_vector(txt_data)

    def __len__(self) -> int:
        return int(len(self.X) / self.seq_length - 1)

    def __getitem__(self, index) -> tuple:
        start_idx = index * self.seq_length
        end_idx = (index + 1) * self.seq_length
        X = torch.tensor(self.X[start_idx:end_idx]).float()
        y = torch.tensor(self.X[start_idx + 1:end_idx + 1]).float()
        return X, y

    def string_to_vector(self, name: str) -> list[int]:
        return [self.char_to_idx[ch] for ch in name]

    def vector_to_string(self, vector: list[int]) -> str:
        return ''.join([self.idx_to_char[i] for i in vector])