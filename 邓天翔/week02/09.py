import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


# 1. 生成模拟数据 (与之前相同)
X_numpy = np.random.rand(100, 1) * 10
# 形状为 (100, 1) 的二维数组，其中包含 100 个在 [0, 1) 范围内均匀分布的随机浮点数。

# 对X进行排序以便更好地可视化
X_numpy_sorted = np.sort(X_numpy, axis=0)

# y_numpy = 2 * X_numpy + 1 + np.random.randn(100, 1)
# 构建一个sin函数
y_numpy = np.sin(X_numpy) + np.random.randn(100, 1) * 0.1
X = torch.from_numpy(X_numpy).float() # torch 中 所有
# 的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

class SimpleRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SimpleRegressor, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# 设置三层隐藏层
# 输入为x， 输出为y， 所以输入输出维度为1
# 设置模型参数
input_dim = 1
hidden_dims = [128, 64, 32]  # 更合理的隐藏层结构
output_dim = 1
model = SimpleRegressor(input_dim, hidden_dims, output_dim)

criterion = nn.MSELoss() # 损失函数 内部自带激活函数，softmax
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
num_epochs = 200
for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
    model.train()
    running_loss = 0.0
    # for idx, (inputs, labels) in enumerate(dataloader):
    for idx in range(len(X)):
        optimizer.zero_grad()
        outputs = model(X[idx])
        loss = criterion(outputs, y[idx])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")


    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(X):.4f}")

# 5. 打印最终学到的参数
print("\n训练完成！")
with torch.no_grad():
    # 对排序后的X进行预测以绘制平滑曲线
    X_sorted_tensor = torch.from_numpy(X_numpy_sorted).float()
    y_predicted_sorted = model(X_sorted_tensor)

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_numpy_sorted, y_predicted_sorted, label='Neural Network Model', color='red', linewidth=2)
plt.plot(X_numpy_sorted, np.sin(X_numpy_sorted), label='True sin(x)', color='green', linestyle='--', linewidth=2)

plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
