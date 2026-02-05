import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

'''
RNN、LSTM、GRU文本分类对比
RNN Loss: 2.3720
LSTM Loss: 0.6564
GRU  Loss: 0.2007
'''
dataset = pd.read_csv('../Week01/dataset.csv', sep='\t', header=None)
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

# 自定义数据集
class CharDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, index):
        text = self.texts[index]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long),  self.labels[index]
    
class Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, model_name):
        super(Classifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

        self.model_name = model_name

    def forward(self, x):
        embedded = self.embedding(x)
        if self.model_name == 'lstm':
            lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
            out = self.fc(hidden_state.squeeze(0)) # batch_size * output_dim
            return out
        elif self.model_name == 'gru':
            output, h_n = self.gru(embedded)
            out = self.fc(h_n.squeeze(0))
            return out
        else:
            output, h_n = self.rnn(embedded) # h_n.shape = (1*1, batch_size, hidden_size)
            out = self.fc(h_n.squeeze(0))
            return out


        

# --- Training and Prediction ---
lstm_dataset = CharDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

models = {
    'RNN': Classifier(vocab_size, embedding_dim, hidden_dim, output_dim, 'rnn'),
    'lstm': Classifier(vocab_size, embedding_dim, hidden_dim, output_dim, 'lstm'),
    'gru': Classifier(vocab_size, embedding_dim, hidden_dim, output_dim, 'gru')
}

num_epochs = 4
for name, model in models.items():
    print(f"开始{name}模型进行文本分类==================================================")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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

def classify_text_lstm(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

index_to_label = {i: label for label, i in label_to_index.items()}

for name, model in models.items():
    print(f"查看{name}文本分类结果：==========================================")
    new_text = "帮我导航到北京"
    predicted_class = classify_text_lstm(new_text, model, char_to_index, max_len, index_to_label)
    print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

    new_text_2 = "查询明天北京的天气"
    predicted_class_2 = classify_text_lstm(new_text_2, model, char_to_index, max_len, index_to_label)
    print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")