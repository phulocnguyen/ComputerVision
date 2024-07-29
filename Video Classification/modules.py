import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def conv_out(img_size, conv_layer):
    input = torch.randn(1, conv_layer.in_channels, img_size[0], img_size[1])
    output = conv_layer(input)
    return output.shape[2], output.shape[3]


def multiconv_out(img_size, conv_layers):
    size = img_size
    for layer in conv_layers:
        size = conv_out(size, layer)
    return size


class EncodeCNN(nn.Module):
    def __init__(self, img_x, img_y, fc_dim1, fc_dim2, dropout_probs, embedding_dim):
        super(EncodeCNN, self).__init__()

        self.img_x = img_x
        self.img_y = img_y
        self.embedding_dim = embedding_dim
        self.dropout_probs = dropout_probs

        # Number of fully connected layers nodes
        self.fc_dim1 = fc_dim1
        self.fc_dim2 = fc_dim2

        self.padding = (0, 0)
        self.stride = (2, 2)
        self.kernel_size = (3, 3)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(
                5, 5), stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.kernel_size,
                      stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=self.kernel_size,
                      stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(128, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=self.kernel_size,
                      stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.conv_layers = [nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0)),
                            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(
                                3, 3), stride=(2, 2), padding=(0, 0)),
                            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(
                                3, 3), stride=(2, 2), padding=(0, 0)),
                            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))]

        self.conv_output = multiconv_out((img_x, img_y), self.conv_layers)

        self.fc1 = nn.Linear(
            in_features=256 * self.conv_output[0] * self.conv_output[1], out_features=fc_dim1)
        self.fc2 = nn.Linear(in_features=self.fc_dim1,
                             out_features=self.fc_dim2)
        self.fc3 = nn.Linear(in_features=self.fc_dim2,
                             out_features=self.embedding_dim)

    def forward(self, input):
        # input shape = [batch size, seuquence length, channels, height, width]
        embed_seq = []
        for t in range(input.size(1)):  # Extract frame at t time
            # Feed every frame into convolutional layers
            x = self.conv1(input[:, t, :, :, :])
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.view(x.size(0), -1)  # Flatten the tensor into 2D

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=self.dropout_probs)
            x = self.fc3(x)

            embed_seq.append(x)

        embed_seq = torch.stack(embed_seq, dim=0).transpose_(0, 1)

        return embed_seq


class DecodeRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_RNN_layers, RNN_units, fc_dim, dropout_probs, num_classes):
        super(DecodeRNN, self).__init__()
        self.RNN_input_size = embedding_dim
        self.hidden_RNN_layers = hidden_RNN_layers
        self.RNN_units = RNN_units
        self.fc_dim = fc_dim
        self.dropout_probs = dropout_probs
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(input_size=self.RNN_input_size,
                            hidden_size=self.RNN_units,
                            num_layers=self.hidden_RNN_layers,
                            batch_first=True)  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)

        self.fc1 = nn.Linear(in_features=self.RNN_units,
                             out_features=self.fc_dim)
        self.fc2 = nn.Linear(in_features=self.fc_dim,
                             out_features=self.num_classes)

    def forward(self, input):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(input, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        # choose RNN_out at the last time step
        x = self.fc1(RNN_out[:, -1, :])
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_probs)
        x = self.fc2(x)

        return x


class DataModule(Dataset):
    def __init__(self, data_path, folder, labels, frames, transform=None):
        self.data_path = data_path
        self.folder = folder
        self.labels = labels
        self.frames = frames
        self.transform = transform

    def len(self):
        return len(self.labels)

    def read_frames(self, path, folder, transform):
        frames = []
        for i in self.frames:
            image = Image.open(os.path.join(
                path, folder, 'frame{:06d}.jpg'.format(i))).convert('L')
            if transform is not None:
                image = transform(image)
            frames.append(image.squeeze(0))

        frames = torch.stack(frames, dim=0)

        return frames

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_frames(self.data_path, folder, self.transform).unsqueeze_(
            0)  # (input) spatial images
        # (labels) LongTensor are for int64 instead of FloatTensor
        y = torch.LongTensor([self.labels[index]])

        # print(X.shape)
        return X, y
