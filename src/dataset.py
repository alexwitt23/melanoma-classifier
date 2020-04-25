""" Custom dataset to define how images are loaded 
for training. """

import pathlib

import torch
import cv2
import albumentations


def training_augmentations(width: int, height: int) -> albumentations.Compose:
    """ Collection of augmentations to perform while training. """
    augmentations = [
        albumentations.Resize(height, width),
        albumentations.Flip(),
        albumentations.RandomRotate90(),
        albumentations.ShiftScaleRotate(),
    ]
    return albumentations.Compose(augmentations)


class LesionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: pathlib.Path,
        img_ext: str = "jpg",
        img_width: int = 100,
        img_height: int = 100,
    ) -> None:
        """ Initialize the dataset by passing in a directory
        of images. """
        super().__init__()

        self.imgs = list(data_dir.glob(f"*{img_ext}"))
        assert self.imgs, f"No images found in {data_dir} with extension {img_ext}."

        self.len = len(self.imgs)
        self.transforms = training_augmentations(img_width, img_height)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> None:
        """ This function is called by PyTorch's DataLoader class.
        Define how images are loaded for training. """

        img_path = self.imgs[idx]
        img = cv2.imread(str(img_path))
        assert img is not None, f"Trouble reading {img_path}!"

        # Perform augmentations
        img_tensor = torch.from_numpy(self.transforms(image=img)["image"])

        # HWC-> CHW
        return img_tensor.permute(2, 0, 1)
