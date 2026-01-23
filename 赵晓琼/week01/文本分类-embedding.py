import torch
import torch.nn as nn
import numpy as np
import random
import json
import pandas as pd
from collections import Counter
import jieba
from sklearn.model_selection import train_test_split

class TextClassificationModel(nn.Module):
    def __init__(self, vocab, vocab_dim, hidden_size, num_classes, sentence_length):
        super(TextClassificationModel, self).__init__()
        # embedding层
        self.embedding = nn.Embedding(len(vocab), vocab_dim, padding_idx = 0)
        self.pooling = nn.AvgPool1d(sentence_length) # AvgPool1d要求通道在最后一维的前一维
        self.linear = nn.Linear(vocab_dim, num_classes)
        self.loss = nn.functional.cross_entropy
    def forward(self, x, y=None):
        x = self.embedding(x) # x.shape = [30, 20, 128] [batch_size, sentence_length, vocab_dim]
        x = x.permute(0, 2, 1) # 调整维度适配池化层 [batch_size, vocab_dim, sentence_length]
        x = self.pooling(x) # 池化后：[batch_size, vocab_dim, 1]
        x = x.squeeze(dim=-1) # 压缩维度：去掉最后一维的1 压缩后:[batch_size, vocab_dim]
        y_pred = self.linear(x) # 全连接层预测 预测值[batch_size, num_classes]
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

# 数据加载
dataset = pd.read_csv('dataset.csv', sep='\t', header=None, nrows=10000)

# jieba分词
dataset['cut_words'] = dataset[0].apply(lambda text: [w for w in jieba.lcut(str(text).strip())]) # dataset[0].apply对数据集中的每一个元素批量应用自定义函数
# 构建词汇表
all_words = []
for words in dataset['cut_words']:
    all_words.extend(words)

word_count = Counter(all_words)

vocab = { "<PAD>": 0 }
for word, count in word_count.most_common(2000): # 保留前2000个高频词
    vocab[word] = len(vocab)

# 标签编码
label2id = {label: idx for idx, label in enumerate(dataset[1].unique())}
dataset['label_id'] = dataset[1].map(label2id)
num_classes = len(label2id)

# 文本转数字序列
sentence_length = 20
def text2seq(text_words):
    # 词转索引，不足补0，超过截断
    seq = [vocab.get(word, 0) for word in text_words[:sentence_length]]
    if len(seq) < sentence_length:
        seq += [0] * (sentence_length - len(seq))
    return seq

dataset['seq'] = dataset['cut_words'].apply(text2seq)

# 划分数据集并转Tensor
X = torch.tensor(dataset['seq'].tolist(), dtype=torch.long)
Y = torch.tensor(dataset['label_id'].tolist(), dtype=torch.long)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# 初始化模型并训练
vocab_dim = 128
hidden_size = 128
sentence_length = 20
model = TextClassificationModel(vocab, vocab_dim, hidden_size, num_classes, sentence_length)


# 训练
def main():
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    batch_size = 32
    epoch_num = 30
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for i in range(0, len(X_train), batch_size):
            # start = batch_index * batch_size
            # end = (batch_index + 1) * batch_size
            x_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            loss = model(x_batch, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        # avg_loss = np.mean(watch_loss)
        print(f"epoch{epoch+1}/{epoch_num},Loss:{loss.item():.4f}")
        evaluate(model)

# 模型评估
def evaluate(model):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_id = torch.argmax(y_pred, dim=1)
        accuracy = (y_pred_id == y_test).float().mean()
        print(f"\n测试集准确率：{accuracy.item():.2f}")

if __name__ == '__main__':
    main()
    id2label = {idx: label for label, idx in label2id.items()}
    while True:
        user_input = input('请输入句子：').strip()
        if user_input == '退出':
            break
        if not user_input:
            print('输入不能为空！')
            continue

        # 处理数据
        # 1.分词
        cut_words = [w for w in jieba.lcut(str(user_input).strip())]
        # 2. 序列化
        seq = text2seq(cut_words)
        seq_tencor = torch.tensor([seq], dtype=torch.long)
        # 3.预测
        model.eval()
        with torch.no_grad():
            pred = model(seq_tencor)
            pred_id = torch.argmax(pred, dim=1).item()
            pred_label = id2label[pred_id]
            print(f'分类结果：{pred_label}\n')

