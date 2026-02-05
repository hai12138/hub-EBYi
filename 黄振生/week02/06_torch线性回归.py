"""
hzs, 2026-01-22, 作业：线性回归

项目目标：
cpu 环境（非深度学习中）下的矩阵运算、向量运算
"""
import numpy as np
import torch


# ==================== 生成数据与预处理参数 ====================
def generate_and_preprocess_data():
    x_numpy = np.random.randn(100, 1) * 10
    y_numpy = 2 * x_numpy +1 +np.random.randn(100, 1)
    x = torch.from_numpy(x_numpy).float()
    y = torch.from_numpy(y_numpy).float()

    print("数据生成完成。")
    print("---" * 10)

    a_float = torch.randn(1, requires_grad=True, dtype=torch.float)
    b_float = torch.randn(1, requires_grad=True, dtype=torch.float)

    print(f"初始参数 a: {a_float.item():.4f}")
    print(f"初始参数 b: {b_float.item():.4f}")

    return x, y, a_float, b_float

# ==================== 训练模型 ====================
def train_model(num_epochs, learning_rate, x, y, a, b):
    for epoch in range(num_epochs):
        y_pred = a * x + b
        loss = torch.mean((y_pred - y)**2)

        if a.grad is not None:
            a.grad.zero_()
        if b.grad is not None:
            b.grad.zero_()

        loss.backward()

        with torch.no_grad():
            a -= learning_rate * a.grad
            b -= learning_rate * b.grad

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


# ==================== 主程序 ====================
def main():
    num_epochs = 1000
    learning_rate = 0.01
    x, y, a, b = generate_and_preprocess_data()
    train_model(num_epochs, learning_rate, x, y, a, b)
    print("\n训练完成！")
    a_learn = a.item()
    b_learn = b.item()
    print(f"拟合的斜率 a: {a_learn:.4f}")
    print(f"拟合的截距 b: {b_learn:.4f}")


# ==================== 程序入口 ====================
if __name__ == "__main__":
    main()