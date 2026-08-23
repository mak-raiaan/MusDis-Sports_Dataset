import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models.multimodal_model import MultimodalModel


def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, coordinates, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        coordinates = coordinates.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, coordinates)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item() * images.size(0)
        all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = accuracy_score(all_labels, all_predictions)
    return epoch_loss, epoch_accuracy


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, coordinates, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            coordinates = coordinates.to(device)
            labels = labels.to(device)
            outputs = model(images, coordinates)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = accuracy_score(all_labels, all_predictions)
    return epoch_loss, epoch_accuracy


def main(data_root="ImageDataset", csv_file="image_coordinates.csv"):
    if not os.path.exists(data_root) or not os.path.exists(csv_file):
        raise FileNotFoundError("Dataset directory or coordinate CSV is missing")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise NotImplementedError(
        "Connect the notebook's dataset transforms and training configuration here."
    )


if __name__ == "__main__":
    main()
