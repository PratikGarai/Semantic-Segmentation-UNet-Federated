import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.transforms as transforms
import pandas as pd

from losses import FocalLoss, mIoULoss
from model import UNet
from dataloader import segDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="path to your dataset")
    parser.add_argument("--test", type=str, help="path to your test dataset")
    parser.add_argument("--meta", type=str, required=True, help="path to your metadata")
    parser.add_argument(
        "--name", type=str, default="unet", help="name to be appended to checkpoints"
    )
    parser.add_argument("--num_epochs", type=int, default=100, help="dnumber of epochs")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument(
        "--loss",
        type=str,
        default="focalloss",
        help="focalloss | iouloss | crossentropy",
    )
    return parser.parse_args()


def acc(label, predicted):
    seg_acc = (y.cpu() == torch.argmax(pred_mask, axis=1).cpu()).sum() / torch.numel(
        y.cpu()
    )
    return seg_acc


def train(
    model=None,
    criterion=None,
    optimizer=None,
    dataloader=None,
    lr_scheduler=None,
    epoch=1,
):
    plot_losses = []
    scheduler_counter = 0
    for epoch in range(N_EPOCHS):
        # training
        model.train()
        loss_list = []
        acc_list = []
        for batch_i, (x, y) in enumerate(dataloader):

            pred_mask = model(x.to(device))
            loss = criterion(pred_mask, y.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.cpu().detach().numpy())
            acc_list.append(acc(y, pred_mask).numpy())

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f)]"
                % (
                    epoch,
                    N_EPOCHS,
                    batch_i,
                    len(dataloader),
                    loss.cpu().detach().numpy(),
                    np.mean(loss_list),
                )
            )
        scheduler_counter += 1
        # testing
        model.eval()
        val_loss_list = []
        val_acc_list = []
        for batch_i, (x, y) in enumerate(test_dataloader):
            with torch.no_grad():
                pred_mask = model(x.to(device))
            val_loss = criterion(pred_mask, y.to(device))
            val_loss_list.append(val_loss.cpu().detach().numpy())
            val_acc_list.append(acc(y, pred_mask).numpy())

        print(
            " epoch {} - loss : {:.5f} - acc : {:.2f} - val loss : {:.5f} - val acc : {:.2f}".format(
                epoch,
                np.mean(loss_list),
                np.mean(acc_list),
                np.mean(val_loss_list),
                np.mean(val_acc_list),
            )
        )
        plot_losses.append([epoch, np.mean(loss_list), np.mean(val_loss_list)])

        compare_loss = np.mean(val_loss_list)
        is_best = compare_loss < min_loss
        if is_best == True:
            scheduler_counter = 0
            min_loss = min(compare_loss, min_loss)
            torch.save(
                model.state_dict(),
                "./saved_models/{}_epoch_{}_{:.5f}.pt".format(
                    args.name, epoch, np.mean(val_loss_list)
                ),
            )

        if scheduler_counter > 5:
            lr_scheduler.step()
            print(f"lowering learning rate to {optimizer.param_groups[0]['lr']}")
            scheduler_counter = 0


def crossvalid(
    model=None,
    criterion=None,
    optimizer=None,
    dataset=None,
    lr_scheduler=None,
    k_fold=5,
    epoch=1,
):
    total_size = len(dataset)
    fraction = 1 / k_fold
    seg = int(total_size * fraction)
    for i in range(k_fold):
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

        train_set = torch.utils.data.dataset.Subset(dataset, train_indices)
        val_set = torch.utils.data.dataset.Subset(dataset, val_indices)

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=50, shuffle=True, num_workers=4
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=50, shuffle=True, num_workers=4
        )
        train(model, criterion, optimizer, train_loader, val_loader, epoch=epoch)


if __name__ == "__main__":
    args = get_args()
    N_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch

    color_shift = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
    blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))

    t = transforms.Compose([color_shift, blurriness])
    dataset = segDataset(args.data, args.meta, training=True, transform=t)
    n_classes = len(dataset.bin_classes) + 1

    print("Number of data : " + str(len(dataset)))

    if not args.test:
        dataset = segDataset(args.data, args.meta, training=True, transform=t)
        n_classes = len(dataset.bin_classes) + 1
        print("Number of data : " + str(len(dataset)))
        test_num = int(0.1 * len(dataset))
        print(f"Test data : {test_num}")
        print(f"Number of classes : {n_classes}")
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [len(dataset) - test_num, test_num], generator=torch.Generator()
        )
        N_DATA, N_TEST = len(train_dataset), len(test_dataset)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
        )
    else:
        dataset = segDataset(args.data, args.meta, training=True, transform=t)
        dataset2 = segDataset(args.test, args.meta, training=False, transform=t)
        n_classes = len(dataset.bin_classes) + 1
        print("Number of train data : " + str(len(dataset)))
        test_num = len(dataset2)
        print(f"Test data : {test_num}")
        print(f"Number of classes : {n_classes}")
        N_DATA, N_TEST = len(dataset), len(dataset2)
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset2, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
        )

    if args.loss == "focalloss":
        criterion = FocalLoss(gamma=3 / 4).to(device)
    elif args.loss == "iouloss":
        criterion = mIoULoss(n_classes=n_classes).to(device)
    elif args.loss == "crossentropy":
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        print("Loss function not found!")

    model = UNet(n_channels=3, n_classes=n_classes, bilinear=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    min_loss = torch.tensor(float("inf"))

    os.makedirs("./saved_models", exist_ok=True)
    crossvalid(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        dataset=dataset,
        lr_scheduler=lr_scheduler,
        k_fold=5,
        epoch=N_EPOCHS,
    )
