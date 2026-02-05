
import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据 (y = 6*sin(x) + 8 + 噪声)
np.random.seed(42)  # 设置随机种子，保证结果可复现
X_numpy = np.random.rand(100, 1) * 10  # x范围[0,10]
y_numpy = 6 * np.sin(X_numpy) + 8 + np.random.randn(100, 1) * 0.5  # 降低噪声，拟合效果更好

# 转换为torch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 创建模型参数 (a对应6，b对应8)
# requires_grad=True 开启梯度计算
a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)

print(f"初始参数 a: {a.item():.4f}")
print(f"初始参数 b: {b.item():.4f}")
print("---" * 10)

# 3. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()  # 回归任务用均方误差
# 调整学习率到更合适的0.01，SGD优化器
optimizer = torch.optim.SGD([a, b], lr=0.01)

# 4. 训练模型
num_epochs = 5000  # 增加训练轮数，保证收敛
for epoch in range(num_epochs):
    # 前向传播：使用torch的sin函数（避免numpy和tensor混用）
    y_pred = a * torch.sin(X) + b

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度（避免累加）
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每500个epoch打印一次损失
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
a_learned = a.item()
b_learned = b.item()
print(f"真实参数：a=6, b=8")
print(f"拟合参数：a={a_learned:.4f}, b={b_learned:.4f}")
print("---" * 10)

# 6. 绘制结果
# 关闭梯度计算（节省资源）
with torch.no_grad():
    y_predicted = a_learned * np.sin(X_numpy) + b_learned  # 用原始numpy数组计算

# 排序数据，让拟合曲线更平滑
sorted_indices = np.argsort(X_numpy, axis=0)
X_sorted = X_numpy[sorted_indices].reshape(-1)
y_pred_sorted = y_predicted[sorted_indices].reshape(-1)

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_sorted, y_pred_sorted,
         label=f'Fit_Result: y = {a_learned:.2f}sin(x) + {b_learned:.2f}',
         color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('sin(x) Data_fit')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
