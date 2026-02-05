import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

x_numpy = np.random.rand(300, 1) * 10
y_numpy = np.sin(x_numpy) + np.random.randn(300, 1) * 0.002

x = torch.from_numpy(x_numpy).float()
y = torch.from_numpy(y_numpy).float()


class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


model = Net(input_size=1, hidden_size1=100, hidden_size2=100, hidden_size3=100, output_size=1)
loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

num_epochs = 10000
model.train()
for epoch in range(num_epochs):

    y_pred = model(x)

    loss = loss_func(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch:{epoch + 1}/{num_epochs} | Loss:{loss.item()}')

model.eval()

with torch.no_grad():
    y_predicted = model(x)
    sorted_idx = torch.argsort(x.squeeze())
    x_sort = x[sorted_idx]
    y_sort = y_predicted[sorted_idx]

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='raw', color='blue', alpha=0.6)
plt.plot(x_sort, y_sort, label=f'y = sin(x)', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()
