import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
number_labels = [label_to_index[label] for label in string_labels]
print(f"label_to_index: {label_to_index}")
print(f"number_labels: {number_labels}")

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40


class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

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


# 3层，损失为0.5820
class SimpleClassifier128(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleClassifier128, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 增加层数 7层，损失为1.3035
class SimpleClassifierNew1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleClassifierNew1, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.fc6 = nn.Linear(128, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        return out


# 增加节点数 3层，损失为0.5867
class SimpleClassifierNew2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleClassifierNew2, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 增加节点数和层数 6层，损失为0.7297
class SimpleClassifierNew3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleClassifierNew3, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(256, 256)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(256, 256)
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(256, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)

        return out


# 只增加节点数,损失为:0.5880
class SimpleClassifierNew4(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleClassifierNew4, self).__init__()
        self.fc1 = nn.Linear(input_dim, 500)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(500, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


char_dataset = CharBoWDataset(texts, number_labels, char_to_index, max_len, vocab_size)

dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

output_dim = len(label_to_index)
# model = SimpleClassifier128(vocab_size, output_dim)
# model = SimpleClassifierNew1(vocab_size, output_dim)
# model = SimpleClassifierNew2(vocab_size, output_dim)
# model = SimpleClassifierNew3(vocab_size, output_dim)
model = SimpleClassifierNew4(vocab_size, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]
    return predicted_label


index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")
