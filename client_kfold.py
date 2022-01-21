import numpy as np
import sys
import argparse
from collections import OrderedDict
import os
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch._C import device
import torchvision.transforms as transforms

import flwr as fl

from losses import FocalLoss, mIoULoss
from model import Custom_Slim_UNet, UNet
from dataloader import segDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rounds = 0

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
        
        global rounds
        print(
            "round {} - epoch {} - fold {} - loss : {:.5f} - acc : {:.2f} - val loss : {:.5f} - val acc : {:.2f}".format(
                rounds,
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
            # torch.save(
            #     self.model.state_dict(),
            #     "./saved_models/{}_{}_epoch_{}_{:.5f}.pt".format(
            #         self.args.name, round epoch, np.mean(val_loss_list)
            #     ),
            # )

        if self.scheduler_counter > 5:
            self.lr_scheduler.step()
            print(f"lowering learning rate to {self.optimizer.param_groups[0]['lr']}")
            self.scheduler_counter = 0




class UNetClient(fl.client.NumPyClient):
    def __init__(self, args) :
        self.t = KFoldTrainer(args)
        os.makedirs("./saved_models", exist_ok=True)
        super().__init__()

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.t.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.t.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        it1 = self.t.model.state_dict().items()
        it2 = state_dict.items()

        l1 = len(it1)
        l2 = len(it2)

        if l1 != l2:
            print(f"{l1} : {l2} length do not match")
        else:
            for i in self.t.model.state_dict():
                if not self.t.model.state_dict()[i].shape == state_dict[i].shape:
                    print(
                        i,
                        self.t.model.state_dict()[i].shape,
                        state_dict[i].shape,
                        "Different",
                    )
        self.t.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        print("Fiting started on Client...")
        global rounds
        self.set_parameters(parameters)
        self.t.train()
        rounds += 1
        return self.get_parameters(), len(self.t.dataset), {}

    def evaluate(self, parameters, config):
        print("Evaluation started on Client...")
        self.set_parameters(parameters)
        # self.t.model.eval()
        val_loss_list = [0]
        val_acc_list = [0]
        # for batch_i, (x, y) in enumerate(self.t.dataset):
        #     with torch.no_grad():
        #         pred_mask = self.t.model(x.to(device))
        #     val_loss = self.t.criterion(pred_mask, y.to(device))
        #     val_loss_list.append(val_loss.cpu().detach().numpy())
        #     val_acc_list.append(acc(y, pred_mask).numpy())

        # print(
        #     " val loss : {:.5f} - val acc : {:.2f}".format(
        #         np.mean(val_loss_list), np.mean(val_acc_list)
        #     )
        # )
        return (
            np.mean(val_loss_list).item(),
            len(self.t.dataset),
            {"accuracy": np.mean(val_acc_list).item()},
        )

if __name__ == "__main__":
    args = get_args()
    fl.client.start_numpy_client(
        server_address="localhost:5000",
        client=UNetClient(args),
    )
