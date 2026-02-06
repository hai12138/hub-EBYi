import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================
# Dataset (char-level)
# ============================================================
class CharDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], char_to_index: Dict[str, int], max_len: int):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        indices = [self.char_to_index.get(ch, 0) for ch in text[: self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# ============================================================
# Model: RNN / LSTM / GRU unified
# ============================================================
class RecurrentClassifier(nn.Module):
    """
    统一封装，确保你只换 rnn_type，就能跑同一套训练/评估流程。
    - rnn_type in {"rnn", "lstm", "gru"}
    - 使用最后一层最后一个 time-step 对应的隐藏状态做分类（等价用 h_n[-1]）
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(
                embedding_dim, hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                nonlinearity="tanh",
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                embedding_dim, hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                embedding_dim, hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"Unknown rnn_type={rnn_type}, should be one of: rnn/lstm/gru")

        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq)
        emb = self.embedding(x)  # (batch, seq, emb)

        if self.rnn_type == "lstm":
            _, (h_n, _) = self.rnn(emb)  # h_n: (num_layers*num_directions, batch, hidden)
        else:
            _, h_n = self.rnn(emb)       # h_n: (num_layers*num_directions, batch, hidden)

        # 取最后一层（如果双向，则最后一层包含正向/反向两个方向）
        # h_n 形状: (num_layers*num_directions, batch, hidden)
        if self.num_directions == 1:
            last = h_n[-1]  # (batch, hidden)
        else:
            # 最后层正向、反向分别在 [-2], [-1]，拼接得到 (batch, 2*hidden)
            last = torch.cat([h_n[-2], h_n[-1]], dim=1)

        return self.fc(last)  # (batch, output_dim)

# ============================================================
# Train / Eval
# ============================================================
@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)

def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 4,
    lr: float = 1e-3,
) -> Tuple[List[float], List[float]]:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_acc_hist, test_acc_hist = [], []

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_acc = evaluate_accuracy(model, train_loader, device)
        test_acc = evaluate_accuracy(model, test_loader, device)
        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)

        print(f"[{model.rnn_type.upper()}] Epoch {ep}/{epochs} | loss={running_loss/len(train_loader):.4f} | "
              f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")

    return train_acc_hist, test_acc_hist

# ============================================================
# Main
# ============================================================
@dataclass
class Config:
    data_path: str = "../lang_week03/dataset.csv"
    max_len: int = 40
    batch_size: int = 32
    embedding_dim: int = 64
    hidden_dim: int = 128
    epochs: int = 4
    lr: float = 1e-3
    test_ratio: float = 0.2
    seed: int = 42
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False
    mode: str = "all"  # all / rnn / lstm / gru

def build_vocab(texts: List[str]) -> Dict[str, int]:
    char_to_index = {"<pad>": 0}
    for t in texts:
        for ch in t:
            if ch not in char_to_index:
                char_to_index[ch] = len(char_to_index)
    return char_to_index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=Config.data_path)
    parser.add_argument("--mode", type=str, default=Config.mode, help="all / rnn / lstm / gru")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--max_len", type=int, default=Config.max_len)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--hidden_dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--embedding_dim", type=int, default=Config.embedding_dim)
    parser.add_argument("--test_ratio", type=float, default=Config.test_ratio)
    parser.add_argument("--num_layers", type=int, default=Config.num_layers)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument("--bidirectional", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    df = pd.read_csv(args.data_path, sep="\t", header=None)
    texts = df[0].astype(str).tolist()
    string_labels = df[1].astype(str).tolist()

    unique_labels = sorted(set(string_labels))
    label_to_index = {lab: i for i, lab in enumerate(unique_labels)}
    y = [label_to_index[lab] for lab in string_labels]
    output_dim = len(unique_labels)

    char_to_index = build_vocab(texts)
    vocab_size = len(char_to_index)

    full_ds = CharDataset(texts, y, char_to_index, args.max_len)

    # Split train/test (stratified 的话更好，这里用简单随机切分)
    idx_all = list(range(len(full_ds)))
    random.shuffle(idx_all)
    split = int(len(idx_all) * (1 - args.test_ratio))
    train_idx = idx_all[:split]
    test_idx = idx_all[split:]

    train_ds = Subset(full_ds, train_idx)
    test_ds = Subset(full_ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    modes = ["rnn", "lstm", "gru"] if args.mode.lower() == "all" else [args.mode.lower()]
    results = {}

    for m in modes:
        model = RecurrentClassifier(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=output_dim,
            rnn_type=m,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
        )
        _, test_hist = train_one_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
        )
        results[m] = test_hist[-1]

    print("\n===== Final Test Accuracy =====")
    for m, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{m.upper():4s}: {acc:.4f}")

if __name__ == "__main__":
    main()
