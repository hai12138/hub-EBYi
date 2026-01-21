import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据 y=sin(x)
X_numpy = np.linspace(0, 4 * np.pi, 100).reshape(-1, 1)  # 使用均匀分布的数据点以获得更好的正弦曲线覆盖
y_numpy = np.sin(X_numpy)
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)


# 2. 定义一个简单的神经网络来拟合非线性函数
class SineNet(torch.nn.Module):
    def __init__(self):
        super(SineNet, self).__init__()
        self.hidden = torch.nn.Linear(1, 30)  # 输入维度为1，隐藏层维度为20
        self.activation = torch.nn.Tanh()  # 使用tanh激活函数
        self.hidden2 = torch.nn.Linear(30, 20)
        self.activation2 = torch.nn.Tanh()
        self.hidden3 = torch.nn.Linear(20, 20)
        self.activation3 = torch.nn.Tanh()
        self.output = torch.nn.Linear(20, 1)  # 输出维度为1

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation2(x)
        x = self.hidden3(x)
        x = self.activation3(x)
        x = self.output(x)
        return x


model = SineNet()

# 初始化参数
learning_rate = 0.01

# 选择损失函数和优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 3. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = criterion(y_pred, y)

    # 反向传播和参数更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

# 4. 绘制结果
with torch.no_grad():
    y_predicted = model(X).numpy()

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='True sin(x)', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label='Fitted curve', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Fitting y = sin(x) using Neural Network')
plt.legend()
plt.grid(True)
plt.show()
