import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CharSeqDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        indices = [self.char_to_index.get(ch, 0) for ch in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


class RecurrentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, cell_type="lstm"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.cell_type = cell_type.lower()

        if self.cell_type == "rnn":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif self.cell_type == "lstm":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("cell_type 必须是 rnn / lstm / gru")

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)  # [B, T, E]

        if self.cell_type == "lstm":
            _, (h_n, _) = self.rnn(embedded)     # h_n: [1, B, H]
        else:
            _, h_n = self.rnn(embedded)          # h_n: [1, B, H]

        last_h = h_n[-1]                          # [B, H]
        logits = self.fc(last_h)                  # [B, C]
        return logits


@torch.no_grad()
def eval_acc(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def train_one(cell_type, train_loader, val_loader, device,
              vocab_size, output_dim, embedding_dim, hidden_dim, lr, epochs):
    print("\n" + "=" * 70)
    print(f"Training {cell_type.upper()} ...")

    model = RecurrentClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, cell_type=cell_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val = eval_acc(model, val_loader, device)
        best_val = max(best_val, val)
        print(f"Epoch {epoch:02d}/{epochs} | loss={running_loss/len(train_loader):.4f} | val_acc={val:.4f}")

    print(f"Best val_acc ({cell_type.upper()}): {best_val:.4f}")
    return best_val


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 读取数据
    dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
    texts = dataset[0].tolist()
    string_labels = dataset[1].tolist()

    # 标签映射
    label_to_index = {label: i for i, label in enumerate(set(string_labels))}
    labels = [label_to_index[l] for l in string_labels]
    output_dim = len(label_to_index)

    # 字符表
    char_to_index = {"<pad>": 0}
    for t in texts:
        for ch in str(t):
            if ch not in char_to_index:
                char_to_index[ch] = len(char_to_index)
    vocab_size = len(char_to_index)

    # 超参
    max_len = 40
    batch_size = 32
    embedding_dim = 64
    hidden_dim = 128
    lr = 0.001
    epochs = 4

    full_ds = CharSeqDataset(texts, labels, char_to_index, max_len)

    # 80/20 划分 train/val
    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 训练并对比
    results = {}
    for cell in ["rnn", "lstm", "gru"]:
        best_val = train_one(cell, train_loader, val_loader, device,
                             vocab_size, output_dim, embedding_dim, hidden_dim, lr, epochs)
        results[cell] = best_val

    print("\n" + "=" * 70)
    print("Final Comparison (Best Validation Accuracy):")
    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{k.upper():>4s}: {v:.4f}")


if __name__ == "__main__":
    main()
