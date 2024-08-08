import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


def data_module():
    data_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    target_transform = transforms.Compose([
        # Resize with nearest interpolation for segmentation labels
        transforms.Resize(
            (512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.tensor(
            np.array(x), dtype=torch.long))  # Convert target to tensor
    ])

    training_data = datasets.OxfordIIITPet(root='./data', split='trainval', transform=data_transform, target_transform=target_transform,
                                           target_types='segmentation', download=True)
    test_data = datasets.OxfordIIITPet(root='./data', split='test', transform=data_transform, target_transform=target_transform,
                                       target_types='segmentation', download=True)

    # Changed shuffle to True for training
    train_loader = DataLoader(training_data, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    return train_loader, test_loader


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes

        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.bottle_neck = self.conv_block(512, 1024)

        self.decoder4 = self.conv_block(1024 + 512, 512)
        self.decoder3 = self.conv_block(512 + 256, 256)
        self.decoder2 = self.conv_block(256 + 128, 128)
        self.decoder1 = self.conv_block(128 + 64, 64)

        self.Downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        # Added mode and align_corners for better upsampling
        self.Upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.output = nn.Conv2d(64, self.num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        return layers

    def forward(self, input):
        encoder1_out = self.encoder1(input)
        out = self.Downsample(encoder1_out)
        encoder2_out = self.encoder2(out)
        out = self.Downsample(encoder2_out)
        encoder3_out = self.encoder3(out)
        out = self.Downsample(encoder3_out)
        encoder4_out = self.encoder4(out)
        out = self.Downsample(encoder4_out)
        bottleneck_out = self.bottle_neck(out)

        decoder4_out = self.Upsample(bottleneck_out)
        decoder4_out = self.decoder4(
            torch.cat([decoder4_out, encoder4_out], dim=1))
        decoder3_out = self.Upsample(decoder4_out)
        decoder3_out = self.decoder3(
            torch.cat([decoder3_out, encoder3_out], dim=1))
        decoder2_out = self.Upsample(decoder3_out)
        decoder2_out = self.decoder2(
            torch.cat([decoder2_out, encoder2_out], dim=1))
        decoder1_out = self.Upsample(decoder2_out)
        decoder1_out = self.decoder1(
            torch.cat([decoder1_out, encoder1_out], dim=1))

        output = self.output(decoder1_out)
        return output


def train(model, train_loader, optimizer, criterion, device, num_epochs):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            outputs = outputs.float()

            labels = labels.long()
            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Print statistics
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    print("Finished training")
    return model


def main():
    print("Creating model")
    model = UNet(num_classes=7)  # Adjust num_classes based on the dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Preparing data for training")
    train_dataloader, test_dataloader = data_module()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 3

    trained_model = train(model, train_dataloader,
                          optimizer, criterion, device, num_epochs)

    torch.save(trained_model.state_dict(), 'UNet.pth')
    print("Model saved to UNet.pth")


if __name__ == '__main__':
    main()
