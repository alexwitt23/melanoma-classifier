Melanoma Classifier 
> A PyTorch model to classify melanoma.

## Setup
The data for this project can be found [here](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000) and [here](https://www.kaggle.com/wanderdust/skin-lesion-analysis-toward-melanoma-detection) from Kaggle. 

The data can be turned into a dataset by running:
```
PYTHONPATH=$(pwd) src/data-prep.py \
    --ham10k_path /path/to/ham \
    --skin_lesion /path/to/skin-lesion \
    --save_dir ~/datasets/melanoma
```
