import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy import ndimage
from glob import glob

from utils.image_utils import get_image_names, get_image_and_mask
from utils.json_utils import get_classes


class segDataset(torch.utils.data.Dataset):
    def __init__(self, root, meta, training, transform=None):
        super(segDataset, self).__init__()
        self.root = root
        self.training = training
        self.transform = transform
        self.BGR_classes, self.bin_classes, self.fs = get_classes(meta)
        self.IMG_NAMES = get_image_names(self.root, self.fs)

    def __getitem__(self, idx):
        img_path = self.IMG_NAMES[idx]
        image, cls_mask = get_image_and_mask(
            img_path, self.BGR_classes, self.bin_classes, self.fs
        )

        if self.training == True:
            if self.transform:
                image = transforms.functional.to_pil_image(image)
                image = self.transform(image)
                image = np.array(image)

            # 90 degree rotation
            if np.random.rand() < 0.5:
                angle = np.random.randint(4) * 90
                image = ndimage.rotate(image, angle, reshape=True)
                cls_mask = ndimage.rotate(cls_mask, angle, reshape=True)

            # vertical flip
            if np.random.rand() < 0.5:
                image = np.flip(image, 0)
                cls_mask = np.flip(cls_mask, 0)

            # horizonal flip
            if np.random.rand() < 0.5:
                image = np.flip(image, 1)
                cls_mask = np.flip(cls_mask, 1)

        image = cv2.resize(image, (512, 512)) / 255.0
        cls_mask = cv2.resize(cls_mask, (512, 512))
        image = np.moveaxis(image, -1, 0)

        return torch.tensor(image).float(), torch.tensor(cls_mask, dtype=torch.int64)

    def __len__(self):
        return len(self.IMG_NAMES)
