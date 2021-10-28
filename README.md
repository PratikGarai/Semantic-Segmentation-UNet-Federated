# U-Net for Semantic Segmentation on Unbalanced Aerial Imagery (Federated Mode)
#### Note: the upstream repository is still under development

## Training

Running the federated server : 
```
python server.py
```

Running the federated client :
```
python client.py --num_epochs 2 --batch 2 --loss focalloss
```

Run atleast 2 clients in two different terminals.