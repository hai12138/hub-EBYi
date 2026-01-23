import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader

dataset = pd.read_csv('dataset.csv', sep='\t', header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40


class CharBowDataset(Dataset):
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

    def __getitem__(self, item):
        return self.bow_vectors[item], self.labels[item]


class SimpleClassifier_1(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(SimpleClassifier_1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


char_dataset = CharBowDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

hidden_dim1 = 128
hidden_dim2 = 128
output_dim = len(label_to_index)
model = SimpleClassifier_1(vocab_size, hidden_dim1, hidden_dim2, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 10

id = 2
print(f'<{id}>层隐藏层 节点数为<{hidden_dim1}> <{hidden_dim2}>')

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

    print(f'Epoch: {(epoch + 1) / num_epochs} Loss: {running_loss / len(dataloader) : .4f}')


def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    # print(bow_vector)
    bow_vector = bow_vector.unsqueeze(0)  # 插入一个维度
    # print(bow_vector)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    # print(predicted_index)
    predicted_index = predicted_index.item()
    # print(predicted_index)
    predicted_label = index_to_label[predicted_index]

    return predicted_label


index_to_label = {i: label for label, i in label_to_index.items()}

new_text = '帮我导航到北京'
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)

new_text1 = '查询明天北京的天气'
predicted_class1 = classify_text(new_text1, model, char_to_index, vocab_size, max_len, index_to_label)
