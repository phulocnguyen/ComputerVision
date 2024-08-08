import os
import torch
import torch.nn as nn
from YOLOdataset import *
from YOLOmodel import *
from torch.utils.data import DataLoader
import tqdm
import argparse

# Define the train function to train the model


def train(device, dataloader, model, optimizer, loss_fn, scaled_anchors):
    model.to(device)

    # Creating a progress bar
    progress_bar = tqdm(dataloader, leave=True)

    # Initializing a list to store the losses
    losses = []

    # Iterating over the training data
    for _, (x, y) in enumerate(progress_bar):
        x = x.to(device)
        y0, y1, y2 = (
            y[0].to(device),
            y[1].to(device),
            y[2].to(device),
        )

        # Getting the model predictions
        outputs = model(x)
        # Calculating the loss at each scale
        loss = (
            loss_fn(outputs[0], y0, scaled_anchors[0])
            + loss_fn(outputs[1], y1, scaled_anchors[1])
            + loss_fn(outputs[2], y2, scaled_anchors[2])
        )

        # Add the loss to the list
        losses.append(loss.item())

        # Reset gradients
        optimizer.zero_grad()

        # Backpropagate the loss
        loss.backward()

        # Optimization step
        optimizer.step()

        # Update progress bar with loss
        mean_loss = sum(losses) / len(losses)
        progress_bar.set_postfix(loss=mean_loss)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available else "cpu"

    ANCHORS = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]

    # Image size
    image_size = 416
    # Grid cell sizes
    s = [image_size // 32, image_size // 16, image_size // 8]

    # Creating the model from YOLOv3 class
    print("Creating the YOLO model")
    model = YOLOv3()

    # Defining the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Defining the loss function
    loss_fn = YOLOLoss()

    # Defining the train dataset
    train_dataset = YOLOdataset(
        csv_file="./data/train.csv",
        image_dir="./data/images/",
        label_dir="./data/labels/",
        anchors=ANCHORS,
        transform=train_transform(image_size)
    )

    # Defining the train data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
    )

    # Scaling the anchors
    scaled_anchors = (
        torch.tensor(ANCHORS) *
        torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)

    # Training the model
    for e in range(1, args.epochs+1):
        print("Epoch:", e)
        train(device, train_loader, model, optimizer, loss_fn, scaled_anchors)


if __name__ == "__main__":
    main()
