import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim as optim


def data_module(batch_size):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_data = datasets.CIFAR100(
        root='./data', train=True, transform=train_transforms, download=True)
    test_data = datasets.CIFAR100(
        root='./data', train=False, transform=test_transforms, download=True)

    valid_size = 0.15

    train_data, valid_data = torch.utils.data.random_split(
        train_data, [1 - valid_size, valid_size])

    train_iterator = data.DataLoader(train_data, batch_size, shuffle=True)

    test_iterator = data.DataLoader(test_data, batch_size)

    valid_iterator = data.DataLoader(valid_data, batch_size)

    return train_iterator, test_iterator, valid_iterator


class PatchEmbeddingLayer(nn.Module):
    def __init__(self, in_channels, patch_size, embedding_dim, num_patches):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten_layer = nn.Flatten(start_dim=1, end_dim=2)
        self.class_token = nn.Parameter(torch.rand(
            1, 1, embedding_dim), requires_grad=True)
        self.positional_embedding = nn.Parameter(torch.rand(
            1, num_patches + 1, embedding_dim), requires_grad=True)

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.permute((0, 2, 3, 1))
        x = self.flatten_layer(x)  # image embedding
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = x + self.positional_embedding
        return x


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, attn_drop=0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.layernorm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multiheadattention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)

    def forward(self, x):
        x = self.layernorm(x)
        output, _ = self.multiheadattention(
            query=x, key=x, value=x, need_weights=False)
        return output


class MultilayerPerceptronBlock(nn.Module):
    def __init__(self, embedding_dim, mlp_size, mlp_dropout):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp_size = mlp_size
        self.mlp_dropout = mlp_dropout
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=mlp_dropout)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        output = self.mlp(x)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim=768, mlp_size=3072, mlp_dropout=0.1, num_heads=12, attn_dropout=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp_size = mlp_size
        self.mlp_dropout = mlp_dropout
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.attention = MultiHeadSelfAttentionBlock(
            embedding_dim=embedding_dim, num_heads=num_heads, attn_drop=attn_dropout)
        self.mlp_block = MultilayerPerceptronBlock(
            embedding_dim, mlp_size, mlp_dropout)

    def forward(self, x):
        x = self.attention(x)
        output = self.mlp_block(x)
        return output


class ViT(nn.Module):
    def __init__(self, img_size=224,
                 in_channels=3,
                 patch_size=16,
                 embedding_dim=768,
                 num_transformer_layers=12,
                 mlp_dropout=0.1,
                 attn_dropout=0.0,
                 mlp_size=3072,
                 num_heads=12,
                 num_classes=10,
                 num_patches=196):
        super().__init__()
        self.patch_embedding_layer = PatchEmbeddingLayer(
            in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim, num_patches=num_patches)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embedding_dim, mlp_size,
                             mlp_dropout, num_heads, attn_dropout)
            for _ in range(num_transformer_layers)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding_layer(x)
        for layer in self.transformer_layers:
            x = layer(x)
        output = self.classifier(x[:, 0])
        return output


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = correct / total
    return val_loss, val_accuracy


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    print('Training complete')
    return model


def main():
    model = ViT()
    BATCH_SIZE = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, test_dataloader, val_dataloader = data_module(
        batch_size=BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 3

    trained_model = train(model, train_dataloader,
                          val_dataloader, criterion, optimizer, num_epochs, device)

    torch.save(trained_model.state_dict(), 'ViT.pth')
    print("Model saved to ViT.pth")


if __name__ == '__main__':
    main()
