import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

'''
构建sin函数
sin函数是非线性的，需要多个神经元和激活函数
'''

# 1.生成模拟数据
X_numpy = np.random.rand(100,1) * 10 # 形状为(100, 1)的二维数组，其中包含100个在[0, 1)范围内均匀分布的随机浮点数

Y_numpy = np.sin(X_numpy)

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(Y_numpy).float()

# 2.直接创建参数张量 w和b
# w = torch.randn(1, requires_grad=True, dtype=torch.float)
# b = torch.randn(1, requires_grad=True, dtype=torch.float)


model = nn.Sequential(
    nn.Linear(1,32), # 包含32个神经元
    nn.Tanh(),
    nn.Linear(32, 1) # 保证输出形状为(N, 1)
)

# 3.定义损失函数和优化器
loss_fn = torch.nn.MSELoss() # 回归任务

optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

# 训练模型
num_epochs = 1000
batch_size = 10
running_loss = 0.0
for epoch in range(num_epochs):
    for i in range(num_epochs//batch_size):
        # y_pred = w * X + b
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 50 == 0:
            print(f"Batch个数{i},当前Batch Loss：{loss.item()}")
    print(f"Epoch[{epoch + 1}/{num_epochs}], Loss:{running_loss/(num_epochs//batch_size):.4f}")
    running_loss = 0.0



x_dense = np.linspace(0, 10, 1000, dtype=np.float32).reshape(-1,1) 
# np.linspace(0, 10, 1000)在0到10的区间内，生成1000个等间距的数值(线性分布)
# .reshape(-1,1) 调整数组维度，把原本的一维数组（形状（1000,））转为二维数组(1000, 1)
y_true = np.sin(x_dense).reshape(-1)
with torch.no_grad():
    x_t = torch.from_numpy(x_dense)
    y_model = model(x_t).numpy().reshape(-1)

plt.close('all')
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(x_dense.reshape(-1), y_true, label='y = sin(x)', color='green', linewidth=2)
ax.plot(x_dense.reshape(-1), y_model, label='model prediction', color='red', linewidth=2)
ax.scatter(X_numpy.reshape(-1), Y_numpy.reshape(-1), label='samples', color='blue', alpha=0.6)
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.legend()
ax.grid(True)
plt.show()



