import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CoordinateImageDataset(Dataset):
    """load images, 33-joint coordinates, and folder-based class labels.

    Args:
        root_folder: directory containing one subdirectory per class.
        csv_file: CSV containing an image name and 66 coordinate values.
        transform: optional image transformation pipeline.
    """

    def __init__(self, root_folder, csv_file, transform=None):
        """index samples without treating the CSV header as data.

        Args:
            root_folder: directory containing class subdirectories.
            csv_file: coordinate CSV with an ``Image`` header.
            transform: optional callable applied to each image.
        """
        self.root_folder = root_folder
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        self.classes = sorted(
            name
            for name in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, name))
        )
        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.classes)
        }
        self.samples = []
        for _, row in self.data.iterrows():
            image_name = row.iloc[0]
            raw_coordinates = row.iloc[1:67].to_numpy(dtype=float)
            if raw_coordinates.size != 66:
                raise ValueError(
                    f"expected 66 coordinates for {image_name}, "
                    f"found {raw_coordinates.size}"
                )
            for class_name in self.classes:
                image_path = os.path.join(
                    root_folder, class_name, image_name
                )
                if os.path.exists(image_path):
                    label = self.class_to_idx[class_name]
                    self.samples.append((image_path, raw_coordinates, label))
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """return one image, normalized joint coordinates, and its label.

        Args:
            index: sample index requested by the data loader.

        Returns:
            A transformed image, 66 normalized coordinate values, and a class
            index. Coordinates are normalized from the original image size.
        """
        image_path, coordinates, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        coordinates = coordinates.reshape(33, 2).copy()
        coordinates[:, 0] /= width
        coordinates[:, 1] /= height
        if self.transform:
            image = self.transform(image)
        coordinates = torch.tensor(
            coordinates.reshape(-1),
            dtype=torch.float32,
        )
        return image, coordinates, label
