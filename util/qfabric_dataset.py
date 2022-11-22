import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset


class SatelliteDataset(Dataset):
    """
    Abstract class.
    """
    def __init__(self, in_c):
        self.in_c = in_c


class QFabricDataset(SatelliteDataset):
    CHANGE_TYPES = [
        "No Change",
        "Residential",
        "Commercial",
        "Industrial",
        "Road",
        "Demolition",
        "Mega Projects",
    ]
    CHANGE_STATUS = ['No Change', 'Prior Construction', 'Greenland',
                     'Land Cleared', 'Excavation', 'Materials Dumped',
                     'Construction Started', 'Construction Midway',
                     'Construction Done', 'Operational']

    # fMoW Stats
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(self, csv_path, is_train=True, t_len=2):
        super(QFabricDataset, self).__init__(in_c=3)
        self.t_len = t_len
        self.is_train = is_train
        self.num_classes = len(self.CHANGE_TYPES)

        self.df = pd.read_csv(csv_path)
        self.num_locs = len(self.df)
        self.dates = self.df[['date:01', 'date:02', 'date:03', 'date:04', 'date:05']].to_numpy()
        self.dates = pd.to_datetime(self.dates.reshape(-1))  # (5*N)
        self.years = np.asarray(self.dates.year).reshape(self.num_locs, 5)
        self.months = np.asarray(self.dates.month).reshape(self.num_locs, 5)
        self.hours = np.asarray(self.dates.hour).reshape(self.num_locs, 5)

        self.image_dirs = self.df[['image:01', 'image:02', 'image:03', 'image:04', 'image:05']]
        self.image_dirs = np.asarray(self.image_dirs)

        self.image_names = self.df[['image-name:01', 'image-name:02', 'image-name:03', 'image-name:04', 'image-name:05']]
        self.image_names = np.asarray(self.image_names)

        self.change_type_dirs = np.asarray(self.df['change-type'])
        self.change_type_names = np.asarray(self.df['change-type-name'])

        self.num_tiles = np.asarray(self.df['num-tiles'])
        self.cum_tiles = np.cumsum(self.num_tiles)

        self.min_year = 2010

        self.normalize = transforms.Normalize(self.mean, self.std)

    def __len__(self):
        return sum(self.df['num-tiles'])

    def get_loc_tile(self, index):
        mask = index < self.cum_tiles
        valid_rows, = np.nonzero(mask)
        row = valid_rows[0]
        tile = index - (self.cum_tiles[row-1] if row-1 >= 0 else 0)
        return row, tile

    def common_transform(self, images, mask):
        to_tensor = transforms.ToTensor()
        tensors = [to_tensor(img) for img in images]
        mask = (to_tensor(mask) * 255).type(torch.long)

        # Random crop
        if self.is_train:
            i, j, h, w = transforms.RandomCrop.get_params(
                mask, output_size=(224, 224))
            tensors = [TF.crop(t, i, j, h, w) for t in tensors]
            mask = TF.crop(mask, i, j, h, w)
        else:
            crop = transforms.CenterCrop(size=(224, 224))
            tensors = [crop(t) for t in tensors]
            mask = crop(mask)

        # Random horizontal flipping
        if self.is_train and random.random() > 0.5:
            tensors = [TF.hflip(t) for t in tensors]
            mask = TF.hflip(mask)

        # Random vertical flipping
        if self.is_train and random.random() > 0.5:
            tensors = [TF.vflip(t) for t in tensors]
            mask = TF.vflip(mask)

        return tensors, mask

    def __getitem__(self, index):
        row, tile_idx = self.get_loc_tile(index)
        assert tile_idx < self.num_tiles[row], \
            f'{index} for loc {row} not factored properly. Tile index {tile_idx}, limit: {self.num_tiles[row]}'

        if self.t_len == 2:
            im_dirs = [self.image_dirs[row][0], self.image_dirs[row][-1]]
            im_names = [self.image_names[row][0], self.image_names[row][-1]]
            years = [self.years[row][0] - self.min_year, self.years[row][-1] - self.min_year]
            months = [self.months[row][0] - 1, self.months[row][-1] - 1]
            hours = [self.hours[row][0], self.hours[row][-1]]
        elif self.t_len == 5:
            im_dirs = list(self.image_dirs[row])
            im_names = list(self.image_names[row])
            years = list(self.years[row] - self.min_year)
            months = list(self.months[row] - 1)
            hours = list(self.hours[row])

        im_paths = []
        for path, name in zip(im_dirs, im_names):
            im_path = os.path.join(path, name + f'.t{tile_idx}.png')
            im_paths.append(im_path)

        change_type_dir = self.change_type_dirs[row]
        change_type_name = self.change_type_names[row].replace('.png', f'.t{tile_idx}.png')
        change_type_path = os.path.join(change_type_dir, change_type_name)

        imgs, label = self.common_transform(
            [Image.open(im_path) for im_path in im_paths],
            Image.open(change_type_path),
        )
        imgs = [self.normalize(img) for img in imgs]
        imgs = torch.stack(imgs, dim=0)  # (t, c, h, w)

        dates = [torch.tensor([yr, mo, hr]) for yr, mo, hr in zip(years, months, hours)]
        dates = torch.stack(dates, dim=0)  # (t, 3)

        return imgs, dates, label.squeeze(0)  # (h, w)
