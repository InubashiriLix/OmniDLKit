import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from MLP import FusionMLP, MLPChannelCfg, MLPHeadCfg  # adjust import as needed

# 1. Prepare KMNIST DataLoader
transform = transforms.ToTensor()
train_dataset = datasets.KMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.KMNIST(
    root="./data", train=False, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 2. Synthetic feature extraction
def extract_features(images):
    mean = images.view(images.size(0), -1).mean(dim=1, keepdim=True)
    std = images.view(images.size(0), -1).std(dim=1, keepdim=True)
    return torch.cat([mean, std], dim=1)


# 3. Model setup
sample_images, _ = next(iter(train_loader))
sample_images_flat = sample_images.view(sample_images.size(0), -1)
sample_features = extract_features(sample_images)

channels = {
    "image": (
        MLPChannelCfg(channel_name="image", hidden=128, act_="ReLU", dropout=0.2),
        sample_images_flat,
    ),
    "features": (
        MLPChannelCfg(channel_name="features", hidden=16, act_="ReLU", dropout=0.1),
        sample_features,
    ),
}
head_cfg = MLPHeadCfg(hidden=64, dropout=0.2, act_="ReLU", n_cls=10)
model = FusionMLP(channels, head_cfg)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 4. Training function
def train(model, loader, optimizer, criterion, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        images_flat = images.view(images.size(0), -1)
        features = extract_features(images).to(device)
        optimizer.zero_grad()
        outputs = model(images_flat, features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total


# 5. Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            images_flat = images.view(images.size(0), -1)
            features = extract_features(images).to(device)
            outputs = model(images_flat, features)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total


# 6. Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 6):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Epoch {epoch}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
