import numpy as np
import sys
import argparse
from collections import OrderedDict

import torch
from torch import nn
from torch._C import device
import torchvision.transforms as transforms

import flwr as fl

from losses import FocalLoss, mIoULoss
from model import UNet
from dataloader import segDataset

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./Semantic segmentation dataset', help='path to your dataset')
    parser.add_argument('--num_epochs', type=int, default=100, help='dnumber of epochs')
    parser.add_argument('--batch', type=int, default=4, help='batch size')
    parser.add_argument('--loss', type=str, default='focalloss', help='focalloss | iouloss | crossentropy')
    return parser.parse_args()

def acc(y, pred_mask):
    seg_acc = (y.cpu() == torch.argmax(pred_mask, axis=1).cpu()).sum() / torch.numel(y.cpu())
    return seg_acc

if __name__ == '__main__':
    args = get_args()
    N_EPOCHS = args.num_epochs
    BACH_SIZE = args.batch

    color_shift = transforms.ColorJitter(.1,.1,.1,.1)
    blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))

    t = transforms.Compose([color_shift, blurriness])
    dataset = segDataset(args.data, training = True, transform= t)

    print('Number of data : '+ str(len(dataset)))

    test_num = int(0.1 * len(dataset))
    print(f'test data : {test_num}')
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset)-test_num, test_num], generator=torch.Generator().manual_seed(101))
    N_DATA, N_TEST = len(train_dataset), len(test_dataset)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BACH_SIZE, shuffle=True, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BACH_SIZE, shuffle=False, num_workers=1)

    if args.loss == 'focalloss':
        criterion = FocalLoss(gamma=3/4).to(device)
    elif args.loss == 'iouloss':
        criterion = mIoULoss(n_classes=6).to(device)
    elif args.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        print('Loss function not found!')


    model = UNet(n_channels=3, n_classes=6, bilinear=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    min_loss = torch.tensor(float('inf'))

    scheduler_counter = 0

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

            if l1!=l2 :
                print(f"{l1} : {l2} length do not match")
            else :
                for i in model.state_dict() :
                    if not model.state_dict()[i].shape == state_dict[i].shape:
                        print(i, model.state_dict()[i].shape, state_dict[i].shape,"Different")

            model.load_state_dict(state_dict, strict=False)

        def fit(self, parameters, config):
            print("Fiting started on Client...")
            self.set_parameters(parameters)
            global scheduler_counter

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
                    acc_list.append(acc(y,pred_mask).numpy())

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
                    break
                scheduler_counter += 1

                print(' epoch {} - loss : {:.5f} - acc : {:.2f}'.format(epoch, np.mean(loss_list), np.mean(acc_list)))
            
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
                val_acc_list.append(acc(y,pred_mask).numpy())
                
            print(' val loss : {:.5f} - val acc : {:.2f}'.format(np.mean(val_loss_list), np.mean(val_acc_list)))

            global min_loss
            compare_loss = np.mean(val_loss_list)
            is_best = compare_loss < min_loss

            if is_best == True:
                scheduler_counter = 0
                min_loss = min(compare_loss, min_loss)
                # torch.save(model.state_dict(), './saved_models/unet_epoch_{}_{:.5f}.pt'.format(epoch,np.mean(val_loss_list)))
            
            if scheduler_counter > 5:
                lr_scheduler.step()
                print(f"lowering learning rate to {optimizer.param_groups[0]['lr']}")
                scheduler_counter = 0

            return np.mean(val_loss_list).item(), len(test_dataloader), {"accuracy":np.mean(val_acc_list).item()}
        
    fl.client.start_numpy_client(
        server_address="localhost:5000", 
        client=UNetClient(), 
    )