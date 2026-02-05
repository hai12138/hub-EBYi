import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False
# 1. 生成sin函数数据（代替原来的线性数据）
np.random.seed (42)
torch.manual_seed (42)

# 生成数据
X_numpy = np.linspace (-3 * np.pi , 3 * np.pi , 1000).reshape (-1 , 1)  # 生成1000个点，范围-3π到3π
y_numpy = np.sin (X_numpy) + np.random.randn (1000 , 1) * 0.1  # sin函数加上一些噪声

# 转换为PyTorch张量
X = torch.from_numpy (X_numpy).float ()
y = torch.from_numpy (y_numpy).float ()

print ("sin函数数据生成完成！")
print (f"数据范围: x ∈ [{X.min ():.2f}, {X.max ():.2f}], y ∈ [{y.min ():.2f}, {y.max ():.2f}]")
print (f"x范围: [-3π, 3π] ≈ [-9.42, 9.42]")
print ("---" * 10)


# 2. 定义多层神经网络（代替简单的线性模型）
class SinFittingNet (nn.Module):
    def __init__(self , input_dim=1 , hidden_dims=[64 , 32] , output_dim=1):
        """
        多层神经网络拟合sin函数
        input_dim: 输入维度 (1维: x坐标)
        hidden_dims: 隐藏层维度列表，例如[64, 32]表示两层隐藏层，第一层64个神经元，第二层32个神经元
        output_dim: 输出维度 (1维: y坐标)
        """
        super (SinFittingNet , self).__init__ ()

        # 创建网络层
        layers = []

        # 输入层
        prev_dim = input_dim

        # 添加隐藏层
        for i , hidden_dim in enumerate (hidden_dims):
            layers.append (nn.Linear (prev_dim , hidden_dim))
            layers.append (nn.ReLU ())  # ReLU激活函数
            layers.append (nn.BatchNorm1d (hidden_dim))  # 批量归一化，帮助训练
            prev_dim = hidden_dim

        # 输出层
        layers.append (nn.Linear (prev_dim , output_dim))

        # 将层组合成序列
        self.model = nn.Sequential (*layers)

        # 打印模型信息
        print (f"神经网络结构: {input_dim} -> {' -> '.join (map (str , hidden_dims))} -> {output_dim}")
        total_params = sum (p.numel () for p in self.parameters ())
        print (f"总参数量: {total_params:,}")

    def forward(self , x):
        return self.model (x)


# 3. 创建并初始化模型
input_dim = 1
hidden_dims = [64 , 32]  # 两层隐藏层，第一层64个神经元，第二层32个神经元
output_dim = 1

model = SinFittingNet (input_dim , hidden_dims , output_dim)
print ("---" * 10)

# 4. 定义损失函数和优化器（改用Adam优化器）
loss_fn = nn.MSELoss ()  # 均方误差损失
optimizer = optim.Adam (model.parameters () , lr=0.001)  # Adam优化器，学习率0.001

print (f"优化器: Adam, 学习率: 0.001")
print ("---" * 10)

# 5. 训练模型
num_epochs = 3000
print_loss_interval = 300  # 每300个epoch打印一次损失

train_losses = []  # 记录训练损失

print ("开始训练神经网络拟合sin函数...")
for epoch in range (num_epochs):
    # 前向传播
    y_pred = model (X)

    # 计算损失
    loss = loss_fn (y_pred , y)

    # 反向传播和优化
    optimizer.zero_grad ()
    loss.backward ()
    optimizer.step ()

    # 记录损失
    train_losses.append (loss.item ())

    # 定期打印损失
    if (epoch + 1) % print_loss_interval == 0 or epoch == 0:
        print (f'Epoch [{epoch + 1:4d}/{num_epochs}], Loss: {loss.item ():.6f}')

print ("\n训练完成！")
print (f"最终损失: {loss.item ():.6f}")
print ("---" * 10)

# 6. 模型预测和评估
model.eval ()  # 切换到评估模式
with torch.no_grad ():
    # 生成更密集的点用于绘制平滑曲线
    X_test_numpy = np.linspace (-3 * np.pi , 3 * np.pi , 500).reshape (-1 , 1)
    X_test = torch.from_numpy (X_test_numpy).float ()
    y_pred = model (X_test).numpy ()

    # 计算真实sin函数值（无噪声）
    y_true = np.sin (X_test_numpy)

    # 计算平均绝对误差
    mae = np.mean (np.abs (y_pred - y_true))
    print (f"模型在测试集上的平均绝对误差: {mae:.6f}")

# 7. 可视化结果
fig = plt.figure (figsize=(15 , 10))

# 子图1：原始数据、真实sin函数和模型预测
plt.subplot (2 , 2 , 1)
plt.scatter (X_numpy , y_numpy , alpha=0.3 , s=10 , label='训练数据 (含噪声)' , color='lightblue')
plt.plot (X_test_numpy , y_true , 'k--' , label='真实 sin(x)' , linewidth=2 , alpha=0.7)
plt.plot (X_test_numpy , y_pred , 'r-' , label='神经网络预测' , linewidth=2)
plt.xlabel ('x')
plt.ylabel ('y')
plt.title ('神经网络拟合 sin(x) 函数')
plt.legend ()
plt.grid (True , alpha=0.3)
plt.axhline (y=0 , color='gray' , linestyle='-' , linewidth=0.5 , alpha=0.5)
plt.axvline (x=0 , color='gray' , linestyle='-' , linewidth=0.5 , alpha=0.5)

plt.show ()
