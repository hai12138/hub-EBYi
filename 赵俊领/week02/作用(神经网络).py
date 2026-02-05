'''
作业一：
1、调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化。
当层数添加和节点数增加时，loss会变小，更加趋近0
'''

'''
作业二：
2、调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化
'''
import torch
import torch.nn as nn
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt

X_numpy = np.linspace(-3*np.pi, 3*np.pi, 500).reshape(-1, 1)
y_numpy = np.sin(X_numpy) + 0.05 * np.random.randn(500, 1)
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(MLP, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),

            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),

            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),

            nn.Linear(hidden_size3, output_size)
        )

    def forward(self, x):
        return self.network(x)

# --- 模型参数和实例化 ---
input_size = 1
hidden_size1 = 64
hidden_size2 = 128
hidden_size3 = 64
output_size = 1

# 实例化模型
model = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
num_epochs = 5000
for epoch in range(num_epochs):
    y_pred = model(X)

    loss = loss_fn(y_pred, y) # 计算损失
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
with torch.no_grad():
    model.eval()  # 设置为评估模式
    y_predicted = model(X)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 6))
plt.plot(X_numpy, y_predicted.numpy(), label='神经网络拟合', color='red', linewidth=2)
plt.plot(X_numpy, np.sin(X_numpy), label='真实sin函数', color='green', alpha=0.7)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()