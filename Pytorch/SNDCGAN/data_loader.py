import os

import numpy as np
from PIL import Image

from torch.utils.data.dataset import Dataset


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class CelebAHQ(Dataset):
    def __init__(self,
                 transforms=None,
                 mode='train',
                 sample_num=64,
                 img_dir=None,
                 img_ext='png',
                 preload_filename="data_list.txt"):

        self.transforms = transforms

        img_ext_list = ['jpg', 'png']
        img_ext_list.append(img_ext)

        dataset_dir = img_dir

        data_list = list()
        OVERRIDE = False

        if not OVERRIDE and os.path.exists(os.path.join(dataset_dir, preload_filename)):
            print("Load data list from existing list.")
            with open(os.path.join(dataset_dir, preload_filename), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data_list.append(line.split("\n")[0])

        else:
            print("Creating file list.")
            for path, _, files in os.walk(dataset_dir):
                for filename in files:
                    if filename.split(".")[-1] in img_ext_list:
                        data_list.append(os.path.join(os.path.abspath(path), filename))

            data_list = sorted(data_list)

            with open(os.path.join(dataset_dir, preload_filename), 'w') as f:
                f.writelines([str(data) + "\n" for data in data_list])

        if mode == 'val':
            self.img_dir = data_list[:sample_num]
        else:
            self.img_dir = data_list[sample_num:]

    def __getitem__(self, index):
        img = pil_loader(self.img_dir[index])

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        return len(self.img_dir)
