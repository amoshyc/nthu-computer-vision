import pandas as pd
from PIL import Image
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as tf


class Data(Dataset):
    def __init__(self, csv_path, img_dir, img_size=(720, 1140)):
        super().__init__()
        self.anns = pd.read_csv(csv_path).to_dict('records')
        self.img_dir = img_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]

        img = Image.open(self.img_dir / ann['name'])
        img = img.convert('RGB')

        rawW, rawH = img.size
        dstW, dstH = self.img_size
        img = img.resize(self.img_size)
        img = tf.to_tensor(img) # [3, H, W]

        lbl = torch.tensor(
            [
                [ann['BR_x'], ann['BR_y']],
                [ann['BL_x'], ann['BL_y']],
                [ann['TL_x'], ann['TL_y']],
                [ann['TR_x'], ann['TR_y']],
            ]
        ) # [4, 2]
        lbl = lbl / torch.tensor([rawW, rawH]).float()
        lbl = lbl * torch.tensor([dstW, dstH]).float()

        return img, lbl


root_dir = Path('./assets/hw3/')
data = Data(root_dir / 'train_labels.csv', root_dir / 'train_images')
print(len(data))

img, lbl = data[-10]
img = tf.to_pil_image(img)
lbl = lbl.numpy()

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.imshow(img)
ax.plot(lbl[:, 0], lbl[:, 1], 'r.')
plt.show()
