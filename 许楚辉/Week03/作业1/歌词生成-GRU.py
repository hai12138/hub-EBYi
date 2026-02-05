import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# 1. 数据准备
file_path = 'jaychou_lyrics.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    # lines = set(f.readlines())
    # text = " ".join(lines)
    text = f.read()

# 找出所有的独立字符并创建映射
vocab = sorted(list(set(text)))
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for idx, char in enumerate(vocab)}
vocab_size = len(vocab)

# 转换为整数序列
text_as_int = np.array([char_to_idx[c] for c in text])


# 2. 定义数据集和数据加载器
class LyricsDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.data_size = len(text) - seq_length

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        # 输入序列和目标序列
        # 输入X1 - X10    从反方向开始移
        # 输出X2 - X11    反方向开始移动
        input_seq = self.text[idx:idx + self.seq_length]
        target_seq = self.text[idx + 1:idx + self.seq_length + 1]
        return (torch.tensor(input_seq, dtype=torch.long),
                torch.tensor(target_seq, dtype=torch.long))


# 创建数据集和数据加载器
seq_length = 100
dataset = LyricsDataset(text_as_int, seq_length)
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 3. 定义模型 (GRU)
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(CharRNN, self).__init__()
        self.hidden_dim = hidden_dim

        # 词嵌入层（可以是词向量的载体）， 层输入token 在 vocab 的位置， 次序转换为嵌入的结果
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM 层
        self.lstm = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        # x: [batch_size, seq_length] -> [batch_size, seq_length, embedding_dim]
        embedded = self.embedding(x)

        # [batch_size, seq_length, embedding_dim] -> [batch_size, seq_length, hidden_dim]
        # hidden: (h_n, c_n)
        output, hidden = self.lstm(embedded, hidden)

        # 将输出转换为二维，以便传递给全连接层
        # [batch_size * seq_length, hidden_dim]
        output = output.reshape(-1, self.hidden_dim)

        # logits: [batch_size * seq_length, vocab_size]
        logits = self.fc(output)

        return logits, hidden

    def init_hidden(self, batch_size):
        # 初始化隐藏状态和单元状态
        return torch.zeros(1, batch_size, self.hidden_dim)


# 4. 训练模型
embedding_dim = 32
hidden_dim = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CharRNN(vocab_size, embedding_dim, hidden_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss() # 二分类 / 多分类的损失函数

# 加载已有的模型权重
model_path = 'lyrics_generator_model_gru.pt'
if os.path.exists(model_path):
    print("载入已有的模型权重...")
    model.load_state_dict(torch.load(model_path))
else:
    epochs = 10
    for epoch in range(epochs):
        for i, (inputs, targets) in enumerate(dataloader):
            batch_size = inputs.size(0)
            hidden = model.init_hidden(batch_size)
            hidden = hidden.to(device) # 将隐藏状态移动到设备上

            # # 分离隐藏状态，避免梯度回传到上一个批次
            # hidden = tuple([h.data for h in hidden])

            inputs, targets = inputs.to(device), targets.to(device)

            model.zero_grad()

            logits, hidden = model(inputs, hidden)
            loss = criterion(logits, targets.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # 检查梯度，如果向量长度超过1，就梯度裁剪
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    # 保存模型
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至 {model_path}")


# 5. 歌词生成函数
def generate_text(model, start_string, num_generate=100):
    model.eval()

    # 将起始字符串转换为张量，unsqueeze：在哪一个维度上扩展
    input_eval = torch.tensor([char_to_idx[s] for s in start_string], dtype=torch.long).unsqueeze(0).to(device)

    generated_text = start_string
    hidden = model.init_hidden(1)
    hidden = hidden.to(device)

    with torch.no_grad():
        for _ in range(num_generate):
            # 将输入张量调整为正确的形状 [batch_size, seq_length]
            input_eval = input_eval.view(1, -1)

            logits, hidden = model(input_eval, hidden)

            # 移除序列长度维度，得到 [batch_size, vocab_size]
            # logits = logits.squeeze(1)

            predicted_id = torch.argmax(logits[-1], dim=-1).item()

            # 更新模型输入，使用预测的字符
            input_eval = torch.tensor([[predicted_id]], dtype=torch.long).to(device)

            generated_text += idx_to_char[predicted_id]

    return generated_text


# 6. 开始生成歌词
start_prompt = "你说"
generated_lyrics = generate_text(model, start_string=start_prompt)
print("\n--- 生成的歌词 ---\n")
print(generated_lyrics)
