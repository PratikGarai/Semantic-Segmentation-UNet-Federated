# U-Net for Semantic Segmentation

## Overview

This repo has the code to train and test U-Net for Semantic Segmentation task over images. Contains both conventional as well as Federated Traning using FedAvg algorithm in Flower framework.

## Getting Dataset

```sh
sh getData.sh
```

## Testing commands 


### Inference Testing

```sh
python inference.py --data data/CityScape-Dataset/C1-Vehicle_NoPeople-65 --img data/CityScape-Dataset/C1-Vehicle_NoPeople-65/Image/ulm_000009_000019_leftImg8bit.png --meta data/CityScape-Dataset --checkpoint saved_models/unet_epoch_0_1.67928.pt --ind 0
```

### Federated server

```sh
python server.py > server.txt
```

### Federated Client

```sh
python client.py --data data/CityScape-Dataset/Train/C1-Vehicle_NoPeople-65 --meta data/CityScape-Dataset --num_epochs 50 --loss crossentropy --name client1 > client1.txt
```

```sh
python client.py --data data/CityScape-Dataset/Train/C2-People_NoVehicle-22 --meta data/CityScape-Dataset --num_epochs 50 --loss crossentropy --name client2 > client2.txt
```

```sh
python client.py --data data/CityScape-Dataset/Train/C3-NoVehicle_NoPeople-11 --meta data/CityScape-Dataset --num_epochs 50 --loss crossentropy --name client3 > client3.txt
```
