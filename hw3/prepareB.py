import csv
import random
import shutil
from tqdm import tqdm
from pathlib import Path

random.seed(999)

ccpd_dir = Path('/home/amoshyc/Downloads/ccpd_dataset/')
n_train = 3000
n_test = 3000
out_dir = Path('./assets/hw3/')
train_dir = out_dir / 'train_images'
test_dir = out_dir / 'test_images'
train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

img_paths = sorted(list(ccpd_dir.glob('**/*-*-*.jpg')))
random.shuffle(img_paths)

img_paths = img_paths[: n_train + n_test]
train_paths = img_paths[:n_train]
test_paths = img_paths[n_train:]

train_labels = []
for i, path in enumerate(tqdm(train_paths)):
    name = f'{i:04d}.jpg'
    shutil.copy(str(path), str(train_dir / name))
    lbl = map(float, path.stem.split('-')[3].replace('&', '_').split('_'))
    train_labels.append([name, *lbl])

test_labels = []
for i, path in enumerate(tqdm(test_paths)):
    name = f'{i:04d}.jpg'
    shutil.copy(str(path), str(test_dir / name))
    lbl = map(float, path.stem.split('-')[3].replace('&', '_').split('_'))
    test_labels.append([name, *lbl])

header = ['name', 'BR_x', 'BR_y', 'BL_x', 'BL_y', 'TL_x', 'TL_y', 'TR_x', 'TR_y']
with (out_dir / 'train_labels.csv').open('w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(train_labels)
with (out_dir / 'test_labels.csv').open('w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(test_labels)
