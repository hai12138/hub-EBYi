import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =========================
# Data Loading
# =========================
dataset = pd.read_csv("../lang_week03/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# ⚠️ 用 sorted 保证标签映射稳定（set 的顺序不稳定，会影响可复现实验）
unique_labels = sorted(set(string_labels))
label_to_index = {label: i for i, label in enumerate(unique_labels)}
numerical_labels = [label_to_index[label] for label in string_labels]
index_to_label = {i: label for label, i in label_to_index.items()}

# =========================
# Char-level vocab
# =========================
char_to_index = {'<pad>': 0}
for text in texts:
    for ch in text:
        if ch not in char_to_index:
            char_to_index[ch] = len(char_to_index)

vocab_size = len(char_to_index)
max_len = 40  # 最大输入的文本长度

# =========================
# Dataset
# =========================
class CharDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(ch, 0) for ch in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# =========================
# GRU Model
# =========================
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq)
        embedded = self.embedding(x)          # (batch, seq, emb)
        _, h_n = self.gru(embedded)           # h_n: (num_layers*num_directions, batch, hidden)
        logits = self.fc(h_n[-1])             # (batch, output_dim)
        return logits

# =========================
# Train
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CharDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if idx % 100 == 0:
            print(f"Batch {idx}, loss={loss.item():.4f}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], loss={running_loss / len(dataloader):.4f}")

# =========================
# Inference
# =========================
def classify_text(text, model, char_to_index, max_len, index_to_label, device):
    indices = [char_to_index.get(ch, 0) for ch in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    x = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()
    return index_to_label[pred]

new_text = "帮我导航到北京"
print(f"输入 '{new_text}' 预测为: '{classify_text(new_text, model, char_to_index, max_len, index_to_label, device)}'")

new_text_2 = "查询明天北京的天气"
print(f"输入 '{new_text_2}' 预测为: '{classify_text(new_text_2, model, char_to_index, max_len, index_to_label, device)}'")
