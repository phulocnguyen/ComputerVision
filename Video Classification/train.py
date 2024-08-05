import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim as optim
from modules import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score # type: ignore
import pickle
import argparse


# set path
data_path = "./data/"    # define UCF-101 RGB data path
action_name_path = './UCF101actions.pkl'

# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embedding_dim = 512      # latent dim extracted by 2D CNN
img_x, img_y = 256, 342  # resize video 2d frame size
dropout_probs = 0.0          # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 3
RNN_hidden_dim = 512
RNN_FC_dim = 256

# training parameters
k = 101             # number of target category

# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 29, 1


def train(model, device, train_loader, optimizer, epoch):
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()
        # output has dim = (batch, number of classes)
        output = rnn_decoder(cnn_encoder(X))

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze(
        ).numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         # computed on CPU

        loss.backward()
        optimizer.step()

        # show information

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
            epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    return losses, scores


def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            # (y_pred != output) get the index of the max log-probability
            y_pred = output.max(1, keepdim=True)[1]

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4,
              'pin_memory': True} if torch.cuda.is_available else {}

    transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

    print("Preparing data")
    # load UCF101 actions names
    with open(action_name_path, 'rb') as f:
        action_names = pickle.load(f)

    # convert labels -> category
    le = LabelEncoder()
    le.fit(action_names)

    # convert category -> 1-hot
    action_category = le.transform(action_names).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(action_category)

    actions = []
    fnames = os.listdir(data_path)

    all_names = []
    for f in fnames:
        loc1 = f.find('v_')
        loc2 = f.find('_g')
        actions.append(f[(loc1 + 2): loc2])

        all_names.append(f)

    # list all data files
    all_X = all_names                  # all video file names
    all_y = le.transform(actions)    # all video labels

    train_list, test_list, train_label, test_label = train_test_split(
        all_X, all_y, test_size=0.25, random_state=42)

    train_set = DataModule(data_path, train_list, train_label,
                           selected_frames, transform=transform)
    valid_set = DataModule(data_path, test_list, test_label,
                           selected_frames, transform=transform)

    train_loader = DataLoader(train_set, **params)
    valid_loader = DataLoader(valid_set, **params)

    # Define model
    print("Creating the model")
    CNN_encoder = EncodeCNN(img_x, img_y, CNN_fc_hidden1, CNN_fc_hidden2,
                            dropout_probs=dropout_probs, embedding_dim=CNN_embedding_dim)
    RNN_decoder = DecodeRNN(CNN_embedding_dim, RNN_hidden_layers,
                            RNN_hidden_dim, RNN_FC_dim, dropout_probs, num_classes=k)
    CRNN_params = list(CNN_encoder.parameters()) + \
        list(RNN_decoder.parameters())
    # Define loss and optimizer
    optimizer = optim.Adam(CRNN_params, lr=learning_rate)

    # start training
    print("Start training the model")
    for epoch in range(epochs):
        # train, test model
        train_losses, train_scores = train(
            [CNN_encoder, RNN_decoder], device, train_loader, optimizer, epoch)
        epoch_test_loss, epoch_test_score = validation(
            [CNN_encoder, RNN_decoder], device, optimizer, valid_loader)

    print("Finished training")


if __name__ == "__main__":
    main()

