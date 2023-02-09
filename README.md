# U-Net for Semantic Segmentation

Code For the paper : [MDPI](https://www.mdpi.com/2079-9292/12/4/896) | [Arxiv](https://arxiv.org/abs/2212.02196) <br />
Full Code Implementation (including Knowledge Distillation) available [here](https://github.com/kinshukdua/FedUKD)

## Overview

This repo has the code to train and test U-Net for Semantic Segmentation task over images. Contains both conventional as well as Federated Traning using FedAvg algorithm in Flower framework.

## Getting Datasets

```sh
sh getAllData.sh
```

## Federated Testing commands 


### Inference Testing

```sh
python inference.py --data data/CityScape-Dataset/C1-Vehicle_NoPeople-65 --img data/CityScape-Dataset/C1-Vehicle_NoPeople-65/Image/ulm_000009_000019_leftImg8bit.png --meta data/CityScape-Dataset --checkpoint saved_models/unet_epoch_0_1.67928.pt --ind 0
```

### Federated server

```sh
python server.py > server.txt
```

### Federated Clients (CityScape)

```sh
python client.py --data data/CityScape-Dataset/Train/C1-Vehicle_NoPeople-65 --test data/CityScape-Dataset/Test --meta data/CityScape-Dataset --num_epochs 50 --loss crossentropy --name client1 > client1.txt
```

```sh
python client.py --data data/CityScape-Dataset/Train/C2-People_NoVehicle-22 --test data/CityScape-Dataset/Test --meta data/CityScape-Dataset --num_epochs 50 --loss crossentropy --name client2 > client2.txt
```

```sh
python client.py --data data/CityScape-Dataset/Train/C3-NoVehicle_NoPeople-11 --test data/CityScape-Dataset/Test --meta data/CityScape-Dataset --num_epochs 50 --loss crossentropy --name client3 > client3.txt
```
### Federated Clients (Chennai)

```sh
python client.py --data data/Chennai-Dataset/Train/D1 --meta data/Chennai-Dataset --test data/Chennai-Dataset/Test/T1 --num_epochs 50 --loss crossentropy --name clientCHN1 > clientCHN1.txt
```

```sh
python client.py --data data/Chennai-Dataset/Train/D2 --meta data/Chennai-Dataset --test data/Chennai-Dataset/Test/T2 --num_epochs 50 --loss crossentropy --name clientCHN2 > clientCHN2.txt
```

```sh
python client.py --data data/Chennai-Dataset/Train/D3 --meta data/Chennai-Dataset --test data/Chennai-Dataset/Test/T3 --num_epochs 50 --loss crossentropy --name clientCHN3 > clientCHN3.txt
```

## Unified Testing commands

### Unified Testing on CityScape Dataset

```sh
python train.py --data data/CityScape-Dataset-Unified/Train --test data/CityScape-Dataset-Unified/Test --meta data/CityScape-Dataset-Unified --num_epochs 50 --loss crossentropy --name UnifiedCSP > UnifiedCSP.txt
```

### Unified Testing on Chennai Dataset

```sh 
python train.py --data data/Chennai-Dataset-Unified/Train --meta data/Chennai-Dataset-Unified --test data/Chennai-Dataset-Unified/Test --num_epochs 50 --loss crossentropy --name UnifiedCHN > UnifiedCHN.txt
```
## KFold Unified commands

### Chennai Dataset

```sh
python train_kfold.py --data data/Chennai-Dataset-KFold/ --meta data/Chennai-Dataset-KFold/ --name ChennaiKFold --folds 5 --epochs 10 --batch 1 --loss crossentropy --model Custom_Slim_UNet > UnifiedCHNFolded.txt
```

### Cityscape Dataset

```sh
python train_kfold.py --data data/CityScape-Dataset-KFold/ --meta data/CityScape-Dataset-KFold/ --name CityScapeKFold --folds 5 --epochs 10 --batch 1 --loss crossentropy --model Custom_Slim_UNet > UnifiedCSPFolded.txt
```

## KFold Federated commmands

### Chennai Dataset

```sh
python client_kfold.py --data data/Chennai-Federated-Dataset-KFold/C1 --meta data/Chennai-Federated-Dataset-KFold --folds 5 --epochs 10 --loss crossentropy --batch 1 --model Custom_Slim_UNet --name clientKFoldCHN1 > clientKFoldCHN1.txt
```

```sh
python client_kfold.py --data data/Chennai-Federated-Dataset-KFold/C2 --meta data/Chennai-Federated-Dataset-KFold --folds 5 --epochs 10 --loss crossentropy --batch 1 --model Custom_Slim_UNet --name clientKFoldCHN2 > clientKFoldCHN2.txt
```

```sh
python client_kfold.py --data data/Chennai-Federated-Dataset-KFold/C3 --meta data/Chennai-Federated-Dataset-KFold --folds 5 --epochs 10 --loss crossentropy --batch 1 --model Custom_Slim_UNet --name clientKFoldCHN3 > clientKFoldCHN3.txt
```

### Cityscape Dataset

```sh
python client_kfold.py --data data/CityScape-Federated-Dataset-KFold/C1 --meta data/CityScape-Federated-Dataset-KFold --folds 5 --epochs 10 --loss crossentropy --batch 1 --model Custom_Slim_UNet --name clientKFoldCSP1 > clientKFoldCSP1.txt
```

```sh
python client_kfold.py --data data/CityScape-Federated-Dataset-KFold/C2 --meta data/CityScape-Federated-Dataset-KFold --folds 5 --epochs 10 --loss crossentropy --batch 1 --model Custom_Slim_UNet --name clientKFoldCSP2 > clientKFoldCSP2.txt
```

```sh
python client_kfold.py --data data/CityScape-Federated-Dataset-KFold/C3 --meta data/CityScape-Federated-Dataset-KFold --folds 5 --epochs 10 --loss crossentropy --batch 1 --model Custom_Slim_UNet --name clientKFoldCSP3 > clientKFoldCSP3.txt
```
