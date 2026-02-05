#2、调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
samples = 1000
x = torch.linspace(-5*np.pi, 5*np.pi, samples)
y = torch.sin(x)
y += torch.normal(0,0.1,y.shape)
y = y.reshape(-1,1) #*10 + torch.randn(100)
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return len(self.x)


myDataset = MyDataset(x.reshape(-1,1), y)
loader = DataLoader(dataset=myDataset, batch_size=64, shuffle=True)
a,b = next(iter(loader))
print(a.shape,b.shape) #torch.Size([32, 1]) torch.Size([32])
class Net(torch.nn.Module):
    def __init__(self, input_size,h1, h2, output_size):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, h1)
        self.sig = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(h1, h2)
        self.relu = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(h2, output_size)
    def forward(self, x):
        x = self.fc1(x)
        x = self.sig(x)
        x = self.fc2(x)
        x = self.relu(x)
        return self.fc3(x)


net = Net(1, 64, 128, 1)
loss_fn = torch.nn.MSELoss()
sgd = torch.optim.Adam(net.parameters(), lr=0.001)
epochs = 1000
for epoch in range(epochs):
    net.train()
    for a, b in loader:
        sgd.zero_grad()
        y_pred = net(a)
        # print(y_pred.shape,b.shape)
        loss = loss_fn(y_pred, b )
        loss.backward()
        sgd.step()
    if epoch % 100 == 0:
        print(epoch, loss.item())

# x = x.reshape(-1,1)
# for e in range(10000):
#     net.train()
#     sgd.zero_grad()
#     y_pred = net(x)
#     # print(y_pred.shape,b.shape)
#     loss = loss_fn(y_pred, y)
#     loss.backward()
#     sgd.step()

net.eval()
with torch.no_grad():
    x1 = torch.linspace(-15, 45, 2*samples)
    y1 = net(x1.reshape(-1,1))

print(x1.shape, y1.shape,x.shape,y.shape)
plt.plot(x, y, label='y', color='red', linewidth=2)
# plt.plot(x1, y1, label='y_pred', color='g', linewidth=2)
plt.scatter(x1, y1, label='Raw data', color='blue', alpha=0.6)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()