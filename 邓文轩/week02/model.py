import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):  # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class DynamicMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int], output_dim: int):
        super().__init__()
        layers = []
        # 动态添加隐藏层
        pre_dim = input_dim
        for dim in hidden_layers:
            layers.append(nn.Linear(pre_dim, dim))
            pre_dim = dim
            layers.append(nn.ReLU())
        # 输出层
        layers.append(nn.Linear(pre_dim, output_dim))

        self.model = nn.Sequential(*layers)  # *可以将列表解包成一个个参数传入！

    def forward(self, x):
        return self.model(x)