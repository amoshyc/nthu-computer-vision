import pandas as pd
from PIL import Image
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as tf
from torch.utils.tensorboard import SummaryWriter


class Data(Dataset):
    def __init__(self, csv_path, img_dir, img_size=(720, 1140)):
        super().__init__()
        self.anns = pd.read_csv(csv_path).to_dict('records')
        self.img_dir = img_dir
        self.img_size = img_size

    def __len__(self):
        # return len(self.anns)
        return 200

    def __getitem__(self, idx):
        ann = self.anns[idx]

        img = Image.open(self.img_dir / ann['name'])
        img = img.convert('RGB')

        rawW, rawH = img.size
        dstW, dstH = self.img_size
        img = img.resize((dstW, dstH))
        img = tf.to_tensor(img)  # [3, H, W]

        size = torch.tensor([rawW, rawH]).float()
        lbl = [
            ann['BR_x'],
            ann['BR_y'],
            ann['BL_x'],
            ann['BL_y'],
            ann['TL_x'],
            ann['TL_y'],
            ann['TR_x'],
            ann['TR_y'],
        ]
        lbl = torch.tensor(lbl).view(4, 2) / size
        lbl = lbl / size
        lbl = lbl.view(-1)

        return img, lbl, size


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.regression = nn.Sequential(
            nn.Conv2d(64, 8, (1, 1)),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Sigmoid(),
        )

    def forward(self, img_b):
        feat_b = self.features(img_b)
        pred_b = self.regression(feat_b)
        return pred_b


root_dir = Path('./assets/hw3/')
data = Data(root_dir / 'train_labels.csv', root_dir / 'train_images', img_size=(72, 114))
pivot = len(data) * 4 // 5
train_set = Subset(data, range(0, pivot))
valid_set = Subset(data, range(pivot, len(data)))
train_loader = DataLoader(train_set, 10)
valid_loader = DataLoader(valid_set, 10)

device = 'cpu'
model = Net().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

log_dir = Path('./runs/') / f'{datetime.now():%b.%d %H:%M:%S}'
log_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir)

step = 0 # global step
for epoch in range(10):
    # Training
    model.train()
    for img_b, lbl_b, size_b in tqdm(iter(train_loader)):
        optimizer.zero_grad()
        pred_b = model(img_b)
        loss = criterion(pred_b, lbl_b)
        loss.backward()
        optimizer.step()

        size_b = size_b.view(-1, 1, 2)
        pred_b = pred_b.detach().view(-1, 4, 2) * size_b
        lbl_b = lbl_b.detach().view(-1, 4, 2) * size_b
        mse = ((pred_b - lbl_b) ** 2).sum(dim=2).sqrt().mean()
        writer.add_scalar('loss/train', loss.detach().item(), step)
        writer.add_scalar('mse/train', mse, step)
        step += 1

    # Validation
    model.eval()
    with torch.no_grad():
        for img_b, lbl_b, size_b in tqdm(iter(valid_loader)):
            pred_b = model(img_b)
            loss = criterion(pred_b, lbl_b)
            size_b = size_b.view(-1, 1, 2)
            pred_b = pred_b.detach().view(-1, 4, 2) * size_b
            lbl_b = lbl_b.detach().view(-1, 4, 2) * size_b
            mse = ((pred_b - lbl_b) ** 2).sum(dim=2).sqrt().mean()
            writer.add_scalar('loss/valid', loss.item(), step)
            writer.add_scalar('mse/valid', mse, step)

torch.save({
    'model': model,
}, log_dir / 'model.pth')