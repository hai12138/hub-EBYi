import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

from loader import load_data
from model import DynamicMLP

torch.manual_seed(666)

# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
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

dataloader = load_data(texts, numerical_labels, char_to_index, max_len, vocab_size)

def train(model: nn.Module,num_epochs: int) -> list[float]:
    """
    训练模型并返回对应损失变化
    :param model:
    :param num_epochs:
    :return:
    """
    criterion = nn.CrossEntropyLoss()  # 损失函数 内部自带激活函数，softmax
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # epoch： 将数据集整体迭代训练一次
    # batch： 数据集汇总为一批训练一次
    train_loss = []
    for epoch in range(num_epochs):  # 12000， batch size 100 -》 batch 个数： 12000 / 100
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

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        train_loss.append(epoch_loss)

    return train_loss

index_to_label = {i: label for label, i in label_to_index.items()}

all_losses = {}
input_dim = vocab_size
output_dim = len(label_to_index)

configs = [
    {"name": "Model-1 (2x128)", "hidden_layers": [128, 128]},
    {"name": "Model-2 (3x64)", "hidden_layers": [64, 64, 64]},
    {"name": "Model-3 (1x256)", "hidden_layers": [256]}
]

for config in configs:
    model = DynamicMLP(input_dim,config["hidden_layers"],output_dim)
    loss = train(model,10)
    all_losses[config["name"]] = loss

plt.figure(figsize = (10,6))

styles = [("-", "blue"), ("--", "red"), (":", "green")]

for idx, (model_name, loss) in enumerate(all_losses.items()):
    linestyle, color = styles[idx % len(styles)]
    plt.plot(loss,linestyle = linestyle,color = color,label = model_name)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.title("Loss Comparison of Different Network Structures")
plt.tight_layout()
plt.savefig("作业1_对比模型loss变化.png", dpi=300, bbox_inches="tight")
plt.show()






