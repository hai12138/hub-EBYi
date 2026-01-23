#encoding='utf-8'
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score

# 设立随机数种子，确保结果可复现
torch.manual_seed(10)
np.random.seed(10)

X_numpy = np.random.uniform(-2*np.pi, 2*np.pi, (100,1))
Y_numpy = 3 * np.sin(2*X_numpy+1) + 1 + 0.1*np.random.randn(100,1)

X = torch.from_numpy(X_numpy).float()
Y = torch.from_numpy(Y_numpy).float()

print("数据生成完成。")
print("---" * 20)

# 2. 直接创建参数张量 a 和 b
a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.tensor([1.8], requires_grad=True, dtype=torch.float)
c = torch.randn(1, requires_grad=True, dtype=torch.float)
d = torch.randn(1, requires_grad=True, dtype=torch.float)

print(f"初始参数 a: {a.item():.4f}")
print(f"初始参数 b: {b.item():.4f}")
print(f"初始参数 c: {c.item():.4f}")
print(f"初始参数 d: {d.item():.4f}")
print("---" * 20)

# 3. 定义学习率
learning_rate = 0.01

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播：计算预测值 y_pred
    y_pred = a * torch.sin(b*X+c) + d

    # 手动计算 MSE 损失
    loss = torch.mean((y_pred - Y)**2)

    # 手动反向传播：计算 a 、 b 和 c 的梯度
    # PyTorch 的自动求导会帮我们计算，我们只需要调用 loss.backward()
    # 但在这里，我们手动计算梯度，因此需要确保梯度清零
    if a.grad is not None:
        a.grad.zero_()
    if b.grad is not None:
        b.grad.zero_()
    if c.grad is not None:
        c.grad.zero_()
    if d.grad is not None:
        d.grad.zero_()

    loss.backward()

    # 手动更新参数
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
a_learned = a.item()
b_learned = b.item()
c_learned = c.item()
d_learned = d.item()
print(f"拟合的参数 a: {a_learned:.4f}")
print(f"拟合的参数 b: {b_learned:.4f}")
print(f"拟合的参数 c: {c_learned:.4f}")
print(f"拟合的参数 d: {d_learned:.4f}")
print("---" * 20)

# 对X_numpy进行排序，并获取排序索引
sort_idx = np.argsort(X_numpy, axis=0).flatten()
X_sorted = X_numpy[sort_idx]
Y_sorted = Y_numpy[sort_idx]

# 6. 绘制结果
with torch.no_grad():
    y_predicted = a_learned * np.sin(b_learned * X_sorted +c_learned) + d_learned

plt.figure(figsize=(10, 6))
plt.scatter(X_sorted, Y_sorted, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_sorted, y_predicted, label=f'Model: y = {a_learned:.2f}*sin({b_learned:.2f} * x + {c_learned:.2f}) + {d_learned:.2f}', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()


print('-'*20)
#CNN方法
class SinNet(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size1=32, hidden_size2=64, hidden_size3=128, output_size=1):
        super(SinNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.relu = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(hidden_size2, hidden_size3)
        self.relu = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


# 设立随机数种子，确保结果可复现
torch.manual_seed(10)
np.random.seed(10)

X_numpy = np.random.uniform(-2*np.pi, 2*np.pi, (100000,1))
Y_numpy = 3 * np.sin(2*X_numpy+1) + 1 + 0.1*np.random.randn(100000,1)

X = torch.from_numpy(X_numpy).float()
Y = torch.from_numpy(Y_numpy).float()

print("数据生成完成。")
print("---" * 20)
sin_dataset = TensorDataset(X, Y)
dataloader = DataLoader(sin_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据
model=SinNet(input_size=1, hidden_size1=32, hidden_size2=64, hidden_size3=128, output_size=1)
print(model)
criterion = nn.MSELoss() # 损失函数 内部自带激活函数，softmax
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 10
train_losses = []
for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 100 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    train_losses.append(running_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

print("训练完成\n")
print('-'*20)

# 可视化结果
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# 绘制训练过程中的损失曲线
axes[0].plot(train_losses, label='train_loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('train_loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 生成用于预测的平滑数据
X_smooth = np.linspace(-2 * np.pi, 2 * np.pi, 500).reshape(-1, 1)
X_smooth_tensor = torch.FloatTensor(X_smooth)
model.eval()
with torch.no_grad():
    y_pred_smooth = model(X_smooth_tensor).numpy()

# 绘制真实sin函数和预测结果
y_true_smooth = 3*np.sin(2*X_smooth+1)+1

#在整个区间上的拟合效果
axes[1].scatter(X_numpy, Y_numpy, alpha=0.5, s=10, label='trainData', color='blue')
axes[1].plot(X_smooth, y_true_smooth, 'g-', linewidth=2, label='real sin(x)')
axes[1].plot(X_smooth, y_pred_smooth, 'r-', linewidth=2, label='predict')
axes[1].set_xlabel('x')
axes[1].set_ylabel('sin(x)')
axes[1].set_title('sin')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#  绘制网络结构信息
print(f"\n总参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 计算并显示最终性能指标
with torch.no_grad():
    y_train_pred = model(X).numpy()

train_mse = mean_squared_error(Y_numpy, y_train_pred)
train_r2 = r2_score(Y_numpy, y_train_pred)

print("\n模型性能指标:")
print(f"训练集 MSE: {train_mse:.6f}")
print(f"训练集 R²: {train_r2:.6f}")

