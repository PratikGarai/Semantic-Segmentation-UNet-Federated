import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.transforms as transforms

from losses import FocalLoss, mIoULoss
from model import Custom_Slim_UNet, UNet
from dataloader import segDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="path to your dataset")
    parser.add_argument("--meta", type=str, required=True, help="path to your metadata")
    parser.add_argument(
        "--name", type=str, default="unet", help="name to be appended to checkpoints"
    )
    parser.add_argument("--folds", type=int, default=5, help="number of folds")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument(
        "--loss",
        type=str,
        default="focalloss",
        help="focalloss | iouloss | crossentropy",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="UNet",
        help="UNet | Custom_Slim_UNet",
    )
    return parser.parse_args()


def acc(y, pred_mask):
    seg_acc = (y.cpu() == torch.argmax(pred_mask, axis=1).cpu()).sum() / torch.numel(
        y.cpu()
    )
    return seg_acc


class KFoldTrainer:
    def __init__(self, args):
        self.args = args
        self.folds = args.folds
        self.epochs = args.epochs
        self.BATCH_SIZE = args.batch
        self.min_loss = torch.tensor(float("inf"))

        color_shift = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
        blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))

        t = transforms.Compose([color_shift, blurriness])
        self.dataset = segDataset(args.data, args.meta, training=True, transform=t)
        self.n_classes = len(self.dataset.bin_classes) + 1

        print("Number of data : " + str(len(self.dataset)))

        if args.loss == "focalloss":
            self.criterion = FocalLoss(gamma=3 / 4).to(device)
        elif args.loss == "iouloss":
            self.criterion = mIoULoss(n_classes=self.n_classes).to(device)
        elif args.loss == "crossentropy":
            self.criterion = nn.CrossEntropyLoss().to(device)
        else:
            print("Loss function not found!")

        if args.model == "Custom_Slim_UNet" :
            print("Using custom slim model")
            self.model = Custom_Slim_UNet(n_channels=3, n_classes=self.n_classes, bilinear=False).to(
                device
            )
        else :
            print("Using unet model")
            self.model = UNet(n_channels=3, n_classes=self.n_classes, bilinear=True).to(
                device
            )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.5
        )
        self.scheduler_counter = 0
        self.epoch = 0
        os.makedirs("./saved_models", exist_ok=True)

    def train(self):
        for i in range(self.epochs):
            self.split_and_train(epoch=i)

    def split_and_train(self, epoch):
        total_size = len(self.dataset)
        fraction = 1 / self.folds
        seg = int(total_size * fraction)
        for i in range(self.folds):
            trll = 0
            trlr = i * seg
            vall = trlr
            valr = i * seg + seg
            trrl = valr
            trrr = total_size

            train_left_indices = list(range(trll, trlr))
            train_right_indices = list(range(trrl, trrr))

            train_indices = train_left_indices + train_right_indices
            val_indices = list(range(vall, valr))

            train_set = torch.utils.data.dataset.Subset(self.dataset, train_indices)
            val_set = torch.utils.data.dataset.Subset(self.dataset, val_indices)

            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=self.args.batch, shuffle=True, num_workers=1
            )
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=self.args.batch, shuffle=True, num_workers=1
            )
            self.single_split_train(train_loader, val_loader, fold=i, epoch=epoch)

    def single_split_train(self, train_loader, val_loader, fold, epoch):
        self.model.train()
        loss_list = []
        acc_list = []
        for batch_i, (x, y) in enumerate(train_loader):
            pred_mask = self.model(x.to(device))
            loss = self.criterion(pred_mask, y.to(device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.cpu().detach().numpy())
            acc_list.append(acc(y, pred_mask).numpy())
            sys.stdout.write(
                "\r[Epoch %d/%d] [Fold %d/%d] [Batch %d/%d] [Loss: %f (%f)]"
                % (
                    epoch,
                    self.epochs,
                    fold,
                    self.folds,
                    batch_i,
                    len(train_loader),
                    loss.cpu().detach().numpy(),
                    np.mean(loss_list),
                )
            )

        self.scheduler_counter += 1

        # testing
        self.model.eval()
        val_loss_list = []
        val_acc_list = []
        for batch_i, (x, y) in enumerate(val_loader):
            with torch.no_grad():
                pred_mask = self.model(x.to(device))
            val_loss = self.criterion(pred_mask, y.to(device))
            val_loss_list.append(val_loss.cpu().detach().numpy())
            val_acc_list.append(acc(y, pred_mask).numpy())

        print(
            " epoch {} - fold {} - loss : {:.5f} - acc : {:.2f} - val loss : {:.5f} - val acc : {:.2f}".format(
                epoch,
                fold,
                np.mean(loss_list),
                np.mean(acc_list),
                np.mean(val_loss_list),
                np.mean(val_acc_list),
            )
        )

        compare_loss = np.mean(val_loss_list)
        is_best = compare_loss < self.min_loss
        if is_best == True:
            self.scheduler_counter = 0
            self.min_loss = min(compare_loss, self.min_loss)
            torch.save(
                self.model.state_dict(),
                "./saved_models/{}_epoch_{}_{:.5f}.pt".format(
                    self.args.name, epoch, np.mean(val_loss_list)
                ),
            )

        if self.scheduler_counter > 5:
            self.lr_scheduler.step()
            print(f"lowering learning rate to {self.optimizer.param_groups[0]['lr']}")
            self.scheduler_counter = 0


if __name__ == "__main__":
    args = get_args()
    k = KFoldTrainer(args)
    k.train()
