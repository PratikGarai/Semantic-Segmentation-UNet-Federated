## Command to check image_utils :

1. Go to ./utils
2. Execute the following
```sh
python image_utils.py ../data/cityscape_data/C1-Vehicle_NoPeople-65 ../data/cityscape_data
```

## Singular traning
```sh
python train.py --data data/cityscape_data/C1-Vehicle_NoPeople-65 --meta data/cityscape_data --num_epochs 1
```

## Inference Testing 
```sh 
python inference.py --data data/cityscape_data/C1-Vehicle_NoPeople-65 --img data/cityscape_data/C1-Vehicle_NoPeople-65/Image/ulm_000009_000019_leftImg8bit.png --meta data/cityscape_data --checkpoint saved_models/unet_epoch_0_1.67928.pt --ind 0
```