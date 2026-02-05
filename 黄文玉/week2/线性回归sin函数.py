import torch
import numpy as np
import matplotlib.pyplot as plt

X_numpy = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)  # 0到2π的均匀点，更适合拟合曲线
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(*X_numpy.shape)  # sin(x) + 少量噪声
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("sin函数数据生成完成。")
print("---" * 10)

w1 = torch.randn(1, 32, requires_grad=True, dtype=torch.float)  # 第一层权重
b1 = torch.randn(32, requires_grad=True, dtype=torch.float)     # 第一层偏置
w2 = torch.randn(32, 1, requires_grad=True, dtype=torch.float)  # 第二层权重
b2 = torch.randn(1, requires_grad=True, dtype=torch.float)      # 第二层偏置

print("多层网络参数初始化完成。")
print("---" * 10)

learning_rate = 0.01

num_epochs = 2000
loss_history = []
for epoch in range(num_epochs):
    h = torch.matmul(X, w1) + b1  # 第一层：线性变换
    h_relu = torch.relu(h)        # ReLU激活，引入非线性
    y_pred = torch.matmul(h_relu, w2) + b2  # 第二层：输出

    loss = torch.mean((y_pred - y)**2)
    loss_history.append(loss.item())

    if w1.grad is not None: w1.grad.zero_()
    if b1.grad is not None: b1.grad.zero_()
    if w2.grad is not None: w2.grad.zero_()
    if b2.grad is not None: b2.grad.zero_()

    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        b1 -= learning_rate * b1.grad
        w2 -= learning_rate * w2.grad
        b2 -= learning_rate * b2.grad

    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

with torch.no_grad():
    h = torch.matmul(X, w1) + b1
    h_relu = torch.relu(h)
    y_predicted = torch.matmul(h_relu, w2) + b2
    y_predicted = y_predicted.numpy()

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(X_numpy, np.sin(X_numpy), label='True sin(x)', color='green', linewidth=2)
plt.scatter(X_numpy, y_numpy, label='Noisy data', color='blue', alpha=0.5, s=10)
plt.plot(X_numpy, y_predicted, label='Fitted curve', color='red', linewidth=2)
plt.xlabel('x (0 ~ 2π)')
plt.ylabel('sin(x)')
plt.title('Multi-Layer Network Fitting sin(x)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(loss_history, color='orange', linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Curve')
plt.grid(True)

plt.tight_layout()
plt.show()
