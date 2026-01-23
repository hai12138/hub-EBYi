import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==================== 主要调整 1: 数据生成部分 ====================
# 原代码: 生成线性数据 y = 2x + 1 + 噪声
# 新代码: 生成正弦函数数据 y = sin(x) + 噪声
print("生成正弦函数数据...")
x_numpy = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)  # 生成-2π到2π的1000个点
y_numpy = np.sin(x_numpy) + 0.1 * np.random.randn(1000, 1)  # 正弦函数加噪声

X = torch.from_numpy(x_numpy).float()
y = torch.from_numpy(y_numpy).float()

print(f"数据形状: X: {X.shape}, y: {y.shape}")
print("---" * 10)


# ==================== 主要调整 2: 模型定义部分 ====================
# 原代码: 直接使用两个参数 a 和 b (y = a*x + b)
# 新代码: 定义多层感知机模型，增加非线性能力拟合sin函数
class SinModel(nn.Module):
    def __init__(self, hidden_size=64):
        super(SinModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),  # 输入层: 1维 -> hidden_size维
            nn.ReLU(),  # 激活函数引入非线性
            nn.Linear(hidden_size, hidden_size),  # 隐藏层
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # 输出层: hidden_size维 -> 1维
        )

    def forward(self, x):
        return self.network(x)


# 实例化模型
model = SinModel(hidden_size=64)
print("模型结构:")
print(model)
print(f"参数数量: {sum(p.numel() for p in model.parameters())}")
print("---" * 10)

# ==================== 主要调整 3: 优化器和超参数 ====================
# 原代码: SGD优化器，学习率0.0005
# 新代码: 使用Adam优化器，更适合非线性问题，调整学习率
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 增加训练轮数，因为sin函数拟合更复杂
num_epochs = 2000
print_interval = 100  # 每100轮打印一次

# ==================== 主要调整 4: 训练循环 ====================
print("开始训练正弦函数拟合...")
losses = []  # 记录损失变化

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    # 打印训练进度
    if (epoch + 1) % print_interval == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成!")
print("---" * 10)

# ==================== 主要调整 5: 模型评估和可视化 ====================
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    # 生成更密集的点用于平滑显示拟合曲线
    x_test = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
    x_test_tensor = torch.from_numpy(x_test).float()
    y_pred_tensor = model(x_test_tensor)
    y_pred_numpy = y_pred_tensor.numpy()

# 计算最终损失
final_loss = loss_fn(model(X), y)
print(f"最终训练损失: {final_loss.item():.6f}")

# 增强可视化
plt.figure(figsize=(12, 4))

# 设置中文字体（关键步骤）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统常用字体
# plt.rcParams['font.sans-serif'] = ['PingFang SC']  # macOS 系统常用字体
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # Linux 系统常用字体

# 解决负号('-')显示为方框的问题
plt.rcParams['axes.unicode_minus'] = False
# 子图1: 拟合效果对比
plt.subplot(1, 2, 1)
plt.scatter(x_numpy, y_numpy, alpha=0.3, label='带噪声的训练数据', s=10, color='blue')
plt.plot(x_test, np.sin(x_test), 'g-', linewidth=2, label='真实 sin(x)')
plt.plot(x_test, y_pred_numpy, 'r-', linewidth=2, label='神经网络拟合')
plt.xlabel('x')
plt.ylabel('y')
plt.title('神经网络拟合正弦函数')
plt.legend()
plt.grid(True)

# 子图2: 损失下降曲线
plt.subplot(1, 2, 2)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失下降曲线')
plt.yscale('log')  # 使用对数坐标更好地观察损失下降
plt.grid(True)

plt.tight_layout()
plt.show()
