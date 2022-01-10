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
from model import UNet
from dataloader import segDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, required=True, help="path to your train dataset"
    )
    parser.add_argument("--test", type=str, help="path to your test dataset")
    parser.add_argument("--meta", type=str, required=True, help="path to your metadata")
    parser.add_argument(
        "--name", type=str, default="unet", help="name to be appended to checkpoints"
    )
    parser.add_argument("--num_epochs", type=int, default=100, help="dnumber of epochs")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--save_step", type=int, default=5, help="epochs to skip")
    parser.add_argument(
        "--loss",
        type=str,
        default="focalloss",
        help="focalloss | iouloss | crossentropy",
    )
    return parser.parse_args()


def acc(y, pred_mask):
    seg_acc = (y.cpu() == torch.argmax(pred_mask, axis=1).cpu()).sum() / torch.numel(
        y.cpu()
    )
    return seg_acc


if __name__ == "__main__":
    args = get_args()
    N_EPOCHS = args.num_epochs
    BACH_SIZE = args.batch

    color_shift = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
    blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))

    t = transforms.Compose([color_shift, blurriness])

    if not args.test:
        dataset = segDataset(args.data, args.meta, training=True, transform=t)
        n_classes = len(dataset.bin_classes) + 1
        print("Number of data : " + str(len(dataset)))
        test_num = int(0.1 * len(dataset))
        print(f"Test data : {test_num}")
        print(f"Number of classes : {n_classes}")
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [len(dataset) - test_num, test_num],
            generator=torch.Generator().manual_seed(101),
        )
        N_DATA, N_TEST = len(train_dataset), len(test_dataset)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BACH_SIZE, shuffle=True, num_workers=2
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=BACH_SIZE, shuffle=False, num_workers=1
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
            dataset, batch_size=BACH_SIZE, shuffle=True, num_workers=2
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset2, batch_size=BACH_SIZE, shuffle=False, num_workers=1
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

    scheduler_counter = 0
    round = 0

    class UNetClient(fl.client.NumPyClient):
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in model.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

            it1 = model.state_dict().items()
            it2 = state_dict.items()

            l1 = len(it1)
            l2 = len(it2)

            if l1 != l2:
                print(f"{l1} : {l2} length do not match")
            else:
                for i in model.state_dict():
                    if not model.state_dict()[i].shape == state_dict[i].shape:
                        print(
                            i,
                            model.state_dict()[i].shape,
                            state_dict[i].shape,
                            "Different",
                        )

            model.load_state_dict(state_dict, strict=False)

        def fit(self, parameters, config):
            print("Fiting started on Client...")
            self.set_parameters(parameters)
            global scheduler_counter, round

            os.makedirs("./saved_models", exist_ok=True)

            plot_losses = []
            plot_accuracies = []

            for epoch in range(N_EPOCHS):
                model.train()
                loss_list = []
                acc_list = []
                for batch_i, (x, y) in enumerate(train_dataloader):

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
                            len(train_dataloader),
                            loss.cpu().detach().numpy(),
                            np.mean(loss_list),
                        )
                    )
                scheduler_counter += 1

                if (epoch + 1) % args.save_step == 0:
                    torch.save(
                        model.state_dict(),
                        "./saved_models/{}_{}_epoch_{}_{:.5f}.pt".format(
                            args.name, round, epoch, np.mean(loss_list)
                        ),
                    )
                plot_losses.append([epoch, np.mean(loss_list)])
                plot_accuracies.append([epoch, np.mean(acc_list)])
                print(
                    " epoch {} - loss : {:.5f} - acc : {:.2f}".format(
                        epoch, np.mean(loss_list), np.mean(acc_list)
                    )
                )

            plot_losses = np.array(plot_losses)
            plot_accuracies = np.array(plot_accuracies)

            plt.figure(figsize=(12, 8))
            plt.plot(plot_losses[:, 0], plot_losses[:, 1], color="b", linewidth=4)
            plt.title(args.loss, fontsize=20)
            plt.xlabel("epoch", fontsize=20)
            plt.ylabel("loss", fontsize=20)
            plt.grid()
            plt.savefig(f"loss_plots_{args.name}_{round}.png")

            plt.figure(figsize=(12, 8))
            plt.plot(
                plot_accuracies[:, 0], plot_accuracies[:, 1], color="b", linewidth=4
            )
            plt.title("accuracy", fontsize=20)
            plt.xlabel("epoch", fontsize=20)
            plt.ylabel("accuracy", fontsize=20)
            plt.grid()
            plt.savefig(f"accuracy_plots_{args.name}_{round}.png")

            round += 1
            return self.get_parameters(), len(train_dataloader), {}

        def evaluate(self, parameters, config):
            print("Evaluation started on Client...")
            self.set_parameters(parameters)
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
                " val loss : {:.5f} - val acc : {:.2f}".format(
                    np.mean(val_loss_list), np.mean(val_acc_list)
                )
            )
            return (
                np.mean(val_loss_list).item(),
                len(test_dataloader),
                {"accuracy": np.mean(val_acc_list).item()},
            )

    fl.client.start_numpy_client(
        server_address="localhost:5000",
        client=UNetClient(),
    )
