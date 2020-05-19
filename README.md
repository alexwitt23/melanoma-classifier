Melanoma Classifier 
> A PyTorch model to classify melanoma.

## Setup

Clone this repository: 
```
git clone --recursive https://github.com/alexwitt2399/melanoma-classifier
```
The data for this project can be found 
[here](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000) and 
[here](https://www.kaggle.com/wanderdust/skin-lesion-analysis-toward-melanoma-detection) 
from Kaggle. 

The data can be turned into a dataset by running:
```
PYTHONPATH=$(pwd) src/data-prep.py \
    --ham10k_path /path/to/ham \
    --skin_lesion /path/to/skin-lesion \
    --save_dir ~/datasets/melanoma
```

## Training 

Training can be run like the following: 
```
PYTHONPATH=$(pwd) src/train.py \
    --model_type efficientnet-b0 \
    --dataset_dir ~/datasets/melanoma
```
Feel free to investigate `src/dataset.py` and tweak the augmentations.