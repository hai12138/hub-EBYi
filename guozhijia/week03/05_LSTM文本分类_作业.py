import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

dataset = pd.read_csv("/Users/guozhijia/Documents/八斗/第一周：课程介绍与大模型基础/Week01/dataset.csv", sep="\t", header=None)
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
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# --- NEW LSTM Model Class ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden_state = self.rnn(embedded)
        # hidden_state 形状: (num_layers, batch, hidden_size)
        # 取最后一层的隐藏状态
        out = self.fc(hidden_state[-1])
        return out

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden_state = self.gru(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

# --- Training and Prediction ---
lstm_dataset = CharDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

lstmModel = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
rnnModel = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
gruModel = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
lstmOptimizer = optim.Adam(lstmModel.parameters(), lr=0.001)
rnnOptimizer = optim.Adam(rnnModel.parameters(), lr=0.001)
gruOptimizer = optim.Adam(gruModel.parameters(), lr=0.001)

num_epochs = 4
for epoch in range(num_epochs):
    lstmModel.train()
    lstm_running_loss = 0.0
    rnn_running_loss = 0.0
    gru_running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        lstmOptimizer.zero_grad()
        rnnOptimizer.zero_grad()
        gruOptimizer.zero_grad()

        lstmOutputs = lstmModel(inputs)
        rnnOutputs = rnnModel(inputs)
        gruOutputs = gruModel(inputs)

        lstmLoss = criterion(lstmOutputs, labels)
        rnnLoss = criterion(rnnOutputs, labels)
        gruLoss = criterion(gruOutputs, labels)

        lstmLoss.backward()
        rnnLoss.backward()
        gruLoss.backward()

        lstmOptimizer.step()
        rnnOptimizer.step()
        gruOptimizer.step()

        lstm_running_loss += lstmLoss.item()
        rnn_running_loss += rnnLoss.item()
        gru_running_loss += gruLoss.item()
        if idx % 50 == 0:
            print(f"lstm - Batch 个数 {idx}, 当前Batch Loss: {lstmLoss.item()}")
            print(f"rnn - Batch 个数 {idx}, 当前Batch Loss: {rnnLoss.item()}")
            print(f"gru - Batch 个数 {idx}, 当前Batch Loss: {gruLoss.item()}")

    print(f"lstm - Epoch [{epoch + 1}/{num_epochs}], Loss: {lstm_running_loss / len(dataloader):.4f}")
    print(f"rnn - Epoch [{epoch + 1}/{num_epochs}], Loss: {rnn_running_loss / len(dataloader):.4f}")
    print(f"gru - Epoch [{epoch + 1}/{num_epochs}], Loss: {gru_running_loss / len(dataloader):.4f}")

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

new_text = "帮我导航到北京"
predicted_class = classify_text_lstm(new_text, lstmModel, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' lstm 模型预测为: '{predicted_class}'")
predicted_class = classify_text_lstm(new_text, rnnModel, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' rnn 模型预测为: '{predicted_class}'")
predicted_class = classify_text_lstm(new_text, gruModel, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' gru 模型预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_lstm(new_text_2, lstmModel, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' lstm 模型预测为: '{predicted_class_2}'")
predicted_class_2 = classify_text_lstm(new_text_2, rnnModel, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' rnn 模型预测为: '{predicted_class_2}'")
predicted_class_2 = classify_text_lstm(new_text_2, gruModel, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' lstm 模型预测为: '{predicted_class_2}'")

'''
精度对比：
    lstm - Batch 个数 0, 当前Batch Loss: 2.4506614208221436
    rnn - Batch 个数 0, 当前Batch Loss: 2.4471559524536133
    gru - Batch 个数 0, 当前Batch Loss: 2.584162950515747
    lstm - Batch 个数 50, 当前Batch Loss: 2.307966470718384
    rnn - Batch 个数 50, 当前Batch Loss: 2.3129031658172607
    gru - Batch 个数 50, 当前Batch Loss: 2.2854745388031006
    lstm - Batch 个数 100, 当前Batch Loss: 2.3481345176696777
    rnn - Batch 个数 100, 当前Batch Loss: 2.3178791999816895
    gru - Batch 个数 100, 当前Batch Loss: 2.1698155403137207
    lstm - Batch 个数 150, 当前Batch Loss: 2.3915255069732666
    rnn - Batch 个数 150, 当前Batch Loss: 2.4095494747161865
    gru - Batch 个数 150, 当前Batch Loss: 1.6798814535140991
    lstm - Batch 个数 200, 当前Batch Loss: 2.3304026126861572
    rnn - Batch 个数 200, 当前Batch Loss: 2.3294553756713867
    gru - Batch 个数 200, 当前Batch Loss: 0.9949318170547485
    lstm - Batch 个数 250, 当前Batch Loss: 2.409926414489746
    rnn - Batch 个数 250, 当前Batch Loss: 2.3895199298858643
    gru - Batch 个数 250, 当前Batch Loss: 1.0858474969863892
    lstm - Batch 个数 300, 当前Batch Loss: 2.2501068115234375
    rnn - Batch 个数 300, 当前Batch Loss: 2.297013282775879
    gru - Batch 个数 300, 当前Batch Loss: 0.6306305527687073
    lstm - Batch 个数 350, 当前Batch Loss: 2.329469680786133
    rnn - Batch 个数 350, 当前Batch Loss: 2.292179822921753
    gru - Batch 个数 350, 当前Batch Loss: 0.6442193984985352
    lstm - Epoch [1/4], Loss: 2.3609
    rnn - Epoch [1/4], Loss: 2.3716
    gru - Epoch [1/4], Loss: 1.4738
    lstm - Batch 个数 0, 当前Batch Loss: 2.445793628692627
    rnn - Batch 个数 0, 当前Batch Loss: 2.472468376159668
    gru - Batch 个数 0, 当前Batch Loss: 0.6422803997993469
    lstm - Batch 个数 50, 当前Batch Loss: 2.394357681274414
    rnn - Batch 个数 50, 当前Batch Loss: 2.3911452293395996
    gru - Batch 个数 50, 当前Batch Loss: 0.625209629535675
    lstm - Batch 个数 100, 当前Batch Loss: 2.354841947555542
    rnn - Batch 个数 100, 当前Batch Loss: 2.335533380508423
    gru - Batch 个数 100, 当前Batch Loss: 0.5607890486717224
    lstm - Batch 个数 150, 当前Batch Loss: 2.3313519954681396
    rnn - Batch 个数 150, 当前Batch Loss: 2.413283109664917
    gru - Batch 个数 150, 当前Batch Loss: 0.42913541197776794
    lstm - Batch 个数 200, 当前Batch Loss: 1.8722494840621948
    rnn - Batch 个数 200, 当前Batch Loss: 2.2765493392944336
    gru - Batch 个数 200, 当前Batch Loss: 0.4756487309932709
    lstm - Batch 个数 250, 当前Batch Loss: 1.9800549745559692
    rnn - Batch 个数 250, 当前Batch Loss: 2.285250425338745
    gru - Batch 个数 250, 当前Batch Loss: 0.38963979482650757
    lstm - Batch 个数 300, 当前Batch Loss: 1.7751222848892212
    rnn - Batch 个数 300, 当前Batch Loss: 2.2672886848449707
    gru - Batch 个数 300, 当前Batch Loss: 0.6578813791275024
    lstm - Batch 个数 350, 当前Batch Loss: 1.9790464639663696
    rnn - Batch 个数 350, 当前Batch Loss: 2.3501126766204834
    gru - Batch 个数 350, 当前Batch Loss: 0.4260237216949463
    lstm - Epoch [2/4], Loss: 2.1149
    rnn - Epoch [2/4], Loss: 2.3597
    gru - Epoch [2/4], Loss: 0.4784
    lstm - Batch 个数 0, 当前Batch Loss: 1.4274251461029053
    rnn - Batch 个数 0, 当前Batch Loss: 2.379847526550293
    gru - Batch 个数 0, 当前Batch Loss: 0.22896085679531097
    lstm - Batch 个数 50, 当前Batch Loss: 1.8711376190185547
    rnn - Batch 个数 50, 当前Batch Loss: 2.4284210205078125
    gru - Batch 个数 50, 当前Batch Loss: 0.19024251401424408
    lstm - Batch 个数 100, 当前Batch Loss: 1.4692635536193848
    rnn - Batch 个数 100, 当前Batch Loss: 2.390899419784546
    gru - Batch 个数 100, 当前Batch Loss: 0.14626450836658478
    lstm - Batch 个数 150, 当前Batch Loss: 1.5503205060958862
    rnn - Batch 个数 150, 当前Batch Loss: 2.3864290714263916
    gru - Batch 个数 150, 当前Batch Loss: 0.4637559950351715
    lstm - Batch 个数 200, 当前Batch Loss: 1.3946855068206787
    rnn - Batch 个数 200, 当前Batch Loss: 2.5647735595703125
    gru - Batch 个数 200, 当前Batch Loss: 0.35785868763923645
    lstm - Batch 个数 250, 当前Batch Loss: 1.1507465839385986
    rnn - Batch 个数 250, 当前Batch Loss: 2.4742937088012695
    gru - Batch 个数 250, 当前Batch Loss: 0.3842232823371887
    lstm - Batch 个数 300, 当前Batch Loss: 1.1033155918121338
    rnn - Batch 个数 300, 当前Batch Loss: 2.3687939643859863
    gru - Batch 个数 300, 当前Batch Loss: 0.5766304135322571
    lstm - Batch 个数 350, 当前Batch Loss: 0.918372392654419
    rnn - Batch 个数 350, 当前Batch Loss: 2.3748559951782227
    gru - Batch 个数 350, 当前Batch Loss: 0.18291594088077545
    lstm - Epoch [3/4], Loss: 1.2856
    rnn - Epoch [3/4], Loss: 2.3705
    gru - Epoch [3/4], Loss: 0.3135
    lstm - Batch 个数 0, 当前Batch Loss: 0.8167775273323059
    rnn - Batch 个数 0, 当前Batch Loss: 2.318312644958496
    gru - Batch 个数 0, 当前Batch Loss: 0.1871364414691925
    lstm - Batch 个数 50, 当前Batch Loss: 0.9061517715454102
    rnn - Batch 个数 50, 当前Batch Loss: 2.4865145683288574
    gru - Batch 个数 50, 当前Batch Loss: 0.23659946024417877
    lstm - Batch 个数 100, 当前Batch Loss: 0.6823402643203735
    rnn - Batch 个数 100, 当前Batch Loss: 2.254499673843384
    gru - Batch 个数 100, 当前Batch Loss: 0.18187864124774933
    lstm - Batch 个数 150, 当前Batch Loss: 0.44215720891952515
    rnn - Batch 个数 150, 当前Batch Loss: 2.3312809467315674
    gru - Batch 个数 150, 当前Batch Loss: 0.28459328413009644
    lstm - Batch 个数 200, 当前Batch Loss: 0.5277371406555176
    rnn - Batch 个数 200, 当前Batch Loss: 2.2010228633880615
    gru - Batch 个数 200, 当前Batch Loss: 0.16027995944023132
    lstm - Batch 个数 250, 当前Batch Loss: 0.7471543550491333
    rnn - Batch 个数 250, 当前Batch Loss: 2.2242555618286133
    gru - Batch 个数 250, 当前Batch Loss: 0.4589097499847412
    lstm - Batch 个数 300, 当前Batch Loss: 0.6340905427932739
    rnn - Batch 个数 300, 当前Batch Loss: 2.4204249382019043
    gru - Batch 个数 300, 当前Batch Loss: 0.27981099486351013
    lstm - Batch 个数 350, 当前Batch Loss: 0.5837418437004089
    rnn - Batch 个数 350, 当前Batch Loss: 2.3820316791534424
    gru - Batch 个数 350, 当前Batch Loss: 0.43705520033836365
    lstm - Epoch [4/4], Loss: 0.6345
    rnn - Epoch [4/4], Loss: 2.3303
    gru - Epoch [4/4], Loss: 0.2255
    输入 '帮我导航到北京' lstm 模型预测为: 'Travel-Query'
    输入 '帮我导航到北京' rnn 模型预测为: 'Weather-Query'
    输入 '帮我导航到北京' gru 模型预测为: 'Travel-Query'
    输入 '查询明天北京的天气' lstm 模型预测为: 'Weather-Query'
    输入 '查询明天北京的天气' rnn 模型预测为: 'Weather-Query'
    输入 '查询明天北京的天气' lstm 模型预测为: 'Weather-Query'
'''