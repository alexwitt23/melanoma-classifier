Melanoma Classifier 
> A PyTorch model to classify melanoma.

## Setup

Clone this repository: 
```
git clone --recursive https://github.com/alexwitt2399/melanoma-classifier
```
The data for this project can be found 
[here](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000) andfrom Kaggle. 

The data can be turned into a dataset by running:
```
PYTHONPATH=$(pwd) src/data-prep.py \
    --ham10k_path /path/to/ham \
    --save_dir ~/datasets/melanoma
```

## Training 

Training can be run like the following: 
```
PYTHONPATH=. src/train.py \
    --config configs/efficientnet-b0.yaml 
```
Feel free to investigate `src/dataset.py` and tweak the augmentations.

`configs` contains the yaml definitions for various model training runs.