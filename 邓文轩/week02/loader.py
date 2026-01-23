import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def create_one_bow_vector(self, text):
        tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        tokenized += [0] * (self.max_len - len(tokenized))

        bow_vector = torch.zeros(self.vocab_size)
        for index in tokenized:
            if index != 0:
                bow_vector[index] += 1
        bow_vector = bow_vector.unsqueeze(0)
        return bow_vector

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


def load_data(texts, labels, char_to_index, max_len, vocab_size):
    char_dataset = CharBoWDataset(texts, labels, char_to_index, max_len, vocab_size)  # 读取单个样本
    dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)  # 读取批量数据集 -》 batch数据
    return dataloader