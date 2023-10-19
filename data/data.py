from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config.config import Config

import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2, InterpolationMode


class ImageData(Dataset):
    def __init__(self,
                 config: 'Config',
                 image_dir: str,
                 gray_scale: bool = False,
                 load_all=False) -> None:
        '''
        从文件夹中读取数据

        ### Args:
            - gray_scale: 是否读取灰度图片
            - image_dir: 图片路径
            - load_all: 是否一次性读取到内存
        '''
        self.config = config
        self.image_paths = [f.path for f in os.scandir(image_dir) if os.path.isfile(f) and os.path.splitext(f)[-1].lower() in [".jpg", ".png", ".jpeg", ".bmp"]]
        self.target_size = config.train_image_size
        self.transforms = v2.Compose(
            [v2.Resize((self.target_size, self.target_size), InterpolationMode.BILINEAR),
             v2.ToImageTensor(),
             v2.ConvertImageDtype(torch.float32),
             v2.Normalize([127.5], [127.5])
             ])
        self.load_all = load_all
        if self.load_all:
            self.images = [self.transforms(read_image(path,
                                                      mode=ImageReadMode.RGB if not gray_scale else ImageReadMode.GRAY).float()) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        if self.load_all:
            image = self.images[idx]
        else:
            image = read_image(self.image_paths[idx], mode=ImageReadMode.RGB).float()
            image = self.transforms(image)
        return image.to(self.config.device)
