import sys
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from utils.json_utils import get_classes
# from json_utils import get_classes


def get_image_names(root: str, fs : dict):
    return sorted(glob(root + f'/{fs["images"]}/*.{fs["file_type"]}'))


def get_image_and_mask(img_path: str, color_dict: dict, classes: list, fs: dict):
    img_path = img_path.replace("\\", "/")
    mask_path = img_path.replace(fs["images"], fs["masks"]).replace(fs["image_substr"], fs["mask_substr"])
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    cls_mask = np.zeros(mask.shape)
    for class_name, bgr in color_dict.items():
        cls_mask[mask == bgr] = classes.index(class_name) + 1
    cls_mask = cls_mask[:, :, 0]

    return image, cls_mask


def main():
    color_dict, class_names, fs = get_classes(sys.argv[2])
    image_path = get_image_names(sys.argv[1], fs)[0]
    _, mask = get_image_and_mask(image_path, color_dict, class_names, fs)

    plt.imshow(mask)
    plt.show()


if __name__ == "__main__":
    main()
