# Alzheimer-classification


## Preparation
```
Edit make_dataset_list. In the main function, you can edit your dataset path
$ conda env create --file environment.yaml
```
## Pretrain using SimCLR
```
$ python contrastive_main.py --batch_size 128
```

## Classification
```
$ python main.py --batch_size 128
```
