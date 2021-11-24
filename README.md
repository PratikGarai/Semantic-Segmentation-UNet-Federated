# U-Net for Semantic Segmentation

## Overview

This repo has the code to train and test U-Net for Semantic Segmentation task over images. Contains both conventional as well as Federated Traning using FedAvg algorithm in Flower framework.

## Steps

Put your dataset in the data folder in the following format : 

```
    data
    |---------- Folder 1
    |           |---------- Image
    |           |             |---------- Image1.png 
    |           |             |---------- Image2.png 
    |           |             |---------- Image3.png 
    |           |           
    |           |---------- Mask
    |           |             |---------- Image1.png 
    |           |             |---------- Image2.png 
    |           |             |---------- Image3.png 
    |           |
    |           |---------- classes.json
    |
    |
    |-----------Folder 2
    |           |---------- Image
    |           |             |---------- Image1.png 
    |           |             |---------- Image2.png 
    |           |             |---------- Image3.png 
    |           |           
    |           |---------- Mask
    |           |             |---------- Image1.png 
    |           |             |---------- Image2.png 
    |           |             |---------- Image3.png 
    |           |
    |           |---------- classes.json
```

## Testing commands 

### Singular traning

```sh
python train.py --data data/cityscape_data/Unified_train --meta data/cityscape_data --num_epochs 1
```

### Inference Testing

```sh
python inference.py --data data/cityscape_data/C1-Vehicle_NoPeople-65 --img data/cityscape_data/C1-Vehicle_NoPeople-65/Image/ulm_000009_000019_leftImg8bit.png --meta data/cityscape_data --checkpoint saved_models/unet_epoch_0_1.67928.pt --ind 0
```

### Federated server

```sh
python server.py > server.txt
```

### Federated Client

```sh
python client.py --data data/cityscape_data/C1-Vehicle_NoPeople-65 --meta data/cityscape_data --num_epochs 15 --loss iouloss --name client1 > client1.txt
```

```sh
python client.py --data data/cityscape_data/C2-People_NoVehicle-22 --meta data/cityscape_data --num_epochs 15 --loss iouloss --name client2 > client2.txt
```

```sh
python client.py --data data/cityscape_data/C3-NoVehicle_NoPeople-11 --meta data/cityscape_data --num_epochs 15 --loss iouloss --name client3 > client3.txt
```
