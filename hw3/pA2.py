from pathlib import Path

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


class Data:
    def __init__(self, csv_path):
        super().__init__()
        self.anns = pd.read_csv(csv_path).to_dict('records')

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        x = torch.tensor(ann['x'])
        y = torch.tensor(ann['y'])
        return x, y


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.rand(3, 1) * 0.001)

    def forward(self, xs):
        ps = self.W[0] * xs ** 2 + self.W[1] * xs + self.W[2]
        return ps


data = Data('./assets/hw3/pA2.csv')
loader = DataLoader(data, batch_size=5)

device = 'cpu'
model = Net().to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

history = {
    'loss': [],
    'a': [],
    'b': [],
    'c': [],
}

for epoch in range(50):
    for xs, ys in iter(loader):
        optimizer.zero_grad()
        ps = model(xs)
        loss = criterion(ps, ys)
        loss.backward()
        optimizer.step()

        history['loss'].append(loss.detach().item())
        history['a'].append(model.W[0].item())
        history['b'].append(model.W[1].item())
        history['c'].append(model.W[2].item())

print(model.W)

fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].plot(range(len(history['loss'])), history['loss'])
ax[1].plot(range(len(history['a'])), history['a'])
ax[2].plot(range(len(history['b'])), history['b'])
ax[3].plot(range(len(history['c'])), history['c'])
plt.show()
