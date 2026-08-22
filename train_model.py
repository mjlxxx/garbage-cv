
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights

DATA_DIR = "trashnet/data/dataset-resized" 
BATCH_SIZE = 16
EPOCHS = 20
LR_HEAD = 0.001       #differnt learnign rates
LR_BACKBONE = 0.0001  

#augmentation trainig
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

#loading it two times
full_dataset = datasets.ImageFolder(DATA_DIR)
class_names = full_dataset.classes
print(f"Found classes: {class_names}")
print(f"Total images: {len(full_dataset)}")

train_size = int(0.85 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])

train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)

train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

#pretrained do not move
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

#dont knwo if this is the right technique, freezing first
for param in model.features.parameters():
    param.requires_grad = False

#unfreeze the last layer
for param in model.features[-3:].parameters():
    param.requires_grad = True

model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model = model.to(device)

#differnt weights have differnt importance
from collections import Counter
class_counts = Counter(train_dataset.targets)
total_count = sum(class_counts.values())
class_weights = torch.tensor(
    [total_count / class_counts[i] for i in range(len(class_names))],
    dtype=torch.float32
).to(device)
print(f"Class counts: {dict(class_counts)}")
print(f"Class weights (higher = rarer class): {class_weights.tolist()}")

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
optimizer = optim.Adam([
    {"params": model.features[-3:].parameters(), "lr": LR_BACKBONE},
    {"params": model.classifier.parameters(), "lr": LR_HEAD},
])
#cosine schedule, decreases learnign rateover length
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler.step()
    train_acc = 100 * correct / total

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.3f} "
          f"| Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")
    
    #best versipn
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "trash_classifier.pt")

with open("labels.json", "w") as f:
    json.dump(class_names, f)

print(f"\nDone. Best validation accuracy: {best_val_acc:.1f}%")
print("Saved trash_classifier.pt (best epoch) and labels.json")