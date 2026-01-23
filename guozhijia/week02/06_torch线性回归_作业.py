import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt
import os

# 1. 生成模拟数据 (与之前相同)
X_numpy = np.random.rand(100, 1) * 10
# 形状为 (100, 1) 的二维数组，其中包含 100 个在 [0, 1) 范围内均匀分布的随机浮点数。

y_numpy = np.sin(X_numpy) + np.random.randn(100, 1) * 0.1
X = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 定义神经网络参数
# 第一层参数：输入 1 维 -> 隐藏层 10 维
W1 = torch.randn(1, 10, requires_grad=True)
b1 = torch.randn(10, requires_grad=True)

# 第二层参数：隐藏层 10 维 -> 输出 1 维
W2 = torch.randn(10, 1, requires_grad=True)
b2 = torch.randn(1, requires_grad=True)

print("参数初始化完成。")
print("---" * 10)

# 3. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam([W1, b1, W2, b2], lr=0.01)

# 4. 训练模型
num_epochs = 2000
for epoch in range(num_epochs):
    hidden = X.mm(W1) + b1  
    
    hidden_activated = torch.tanh(hidden)
    
    y_pred = hidden_activated.mm(W2) + b2

    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("\n训练完成！")
print("---" * 10)

# 6. 绘制结果
with torch.no_grad():
    # 为了画出平滑的曲线，我们需要对 X 进行排序
    X_sorted, sorted_indices = torch.sort(X, dim=0)
    
    # 手动计算预测值 (与训练时的逻辑一致)
    h = X_sorted.mm(W1) + b1
    h_act = torch.tanh(h)
    y_predicted = h_act.mm(W2) + b2

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
# 注意：这里画图用的是排序后的 X 和对应的预测值 y
plt.plot(X_sorted.numpy(), y_predicted.numpy(), label='Model Prediction', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)

output_path = os.path.join(os.path.dirname(__file__), 'model_prediction.png')
plt.savefig(output_path)
print(f"模型拟合图已保存至: {output_path}")
plt.show()

'''
loss结果：
    Epoch [100/2000], Loss: 0.2410
    Epoch [200/2000], Loss: 0.1690
    Epoch [300/2000], Loss: 0.1374
    Epoch [400/2000], Loss: 0.1208
    Epoch [500/2000], Loss: 0.1073
    Epoch [600/2000], Loss: 0.0919
    Epoch [700/2000], Loss: 0.0548
    Epoch [800/2000], Loss: 0.0196
    Epoch [900/2000], Loss: 0.0102
    Epoch [1000/2000], Loss: 0.0087
    Epoch [1100/2000], Loss: 0.0085
    Epoch [1200/2000], Loss: 0.0085
    Epoch [1300/2000], Loss: 0.0084
    Epoch [1400/2000], Loss: 0.0084
    Epoch [1500/2000], Loss: 0.0084
    Epoch [1600/2000], Loss: 0.0084
    Epoch [1700/2000], Loss: 0.0084
    Epoch [1800/2000], Loss: 0.0084
    Epoch [1900/2000], Loss: 0.0084
    Epoch [2000/2000], Loss: 0.0084
'''