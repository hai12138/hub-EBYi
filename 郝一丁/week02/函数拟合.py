import torch
import numpy as np
import matplotlib.pyplot as plt

def train_sin_fitting_model(
    num_samples=100,
    x_range=(-2 * np.pi, 2 * np.pi),
    hidden_layers=[32, 32],  # 可配置隐藏层结构
    lr=0.01,
    num_epochs=2000,
    seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. 生成模拟数据 (与之前相同)
    X_numpy = np.random.uniform(x_range[0], x_range[1], (num_samples, 1)).astype(np.float32)
    y_numpy = np.sin(X_numpy).astype(np.float32)

    X = torch.from_numpy(X_numpy)
    y = torch.from_numpy(y_numpy)

    # 2. 定义多层网络
    layers = []
    input_dim = 1
    for hidden_dim in hidden_layers:
        layers.append(torch.nn.Linear(input_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
        input_dim = hidden_dim
    layers.append(torch.nn.Linear(input_dim, 1))
    model = torch.nn.Sequential(*layers)

    # 3. 损失函数和优化器
    # 损失函数仍然是均方误差 (MSE)。
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. 训练循环
    losses = []
    for epoch in range(num_epochs):
        model.train()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # 5. 获取最终预测结果
    model.eval()
    with torch.no_grad():
        y_pred_final = model(X).numpy()

    return X_numpy, y_numpy, y_pred_final, losses

