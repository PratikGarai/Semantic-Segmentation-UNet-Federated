import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

from model import UNet
from dataloader import segDataset

from utils.json_utils import get_classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True, help='path to your image')
    parser.add_argument('--data', type=str, required=True, help='path to your dataset')
    parser.add_argument('--ind', type=str, required=True, help='index to your image in dataset')
    parser.add_argument('--meta', type=str, required=True, help='path to your metadata')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to your model checkpoint')
    return parser.parse_args()


def get_mask_from_real_image(img_path : str, fs : dict):
    return img_path.replace(fs["images"], fs["masks"]).replace(fs["image_substr"], fs["mask_substr"])


if __name__ == '__main__':
    args = get_args()

    _, _, fs = get_classes(args.meta)

    color_shift = transforms.ColorJitter(.1,.1,.1,.1)
    blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))
    t = transforms.Compose([color_shift, blurriness])
    dataset = segDataset(args.data, args.meta, training = False, transform= t)
    n_classes = len(dataset.bin_classes)+1

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    model = UNet(n_channels=3, n_classes=n_classes, bilinear=True).to(device)
    model.load_state_dict(torch.load(args.checkpoint), strict=False)

    real_image = cv2.imread(args.img)
    real_mask = cv2.imread(get_mask_from_real_image(args.img, fs))
    plt.figure("Real image")
    plt.imshow(real_image)
    plt.figure("Real mask")
    plt.imshow(real_mask)

    model.eval()

    for batch_i, (x, y) in enumerate(dataloader):
        with torch.no_grad() :
            processed_image = model(x.to(device))
            break
    
    res = np.argmin(processed_image.numpy()[0], axis=0)
    with open("tmp.txt", "w") as fl:
        for i in res :
            s = ""
            for j in i :
                s += str(j).center(4)
            fl.write(s+"\n")
    
    plt.figure(f"Predicted mask")
    plt.imshow(res)
    plt.show()