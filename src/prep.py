import random 
import pathlib


img_dir = pathlib.Path("/home/alex/datasets/melanoma10k")

imgs = img_dir.glob(f"*.jpg")

train_dir = img_dir / "train"
eval_dir = img_dir / "eval"

train_dir.mkdir(exist_ok=True, parents=True)
eval_dir.mkdir(exist_ok=True, parents=True)

for img in imgs:
    if random.randint(0, 100) < 19:

        img.rename(eval_dir / img.name)

    else:
        img.rename(train_dir / img.name)
