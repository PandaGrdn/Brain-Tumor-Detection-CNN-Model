import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

num_epochs = 10
batch_size = 32
learning_rate = 0.001
num_folds = 5

class BrainTumorDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = datasets.folder.default_loader(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

        if self.transform:
            image = self.transform(image)
        return image, label

def load_data(root_dir):
    data = []
    classes = ['yes', 'no']
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(root_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            data.append((img_path, label))
    return data


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data = load_data('/Users/ethan/Documents/AI_Stuff/AI_Learning_Projects/brain_cancer_detection_stuff/brain_tumor_dataset')

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    total_step = len(train_loader)
    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if images is None or labels is None:
                continue
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

def test_model(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            if images is None or labels is None:
                continue
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

accuracies = []

for fold, (train_index, val_index) in enumerate(kf.split(data)):
    print(f'Fold {fold + 1}')
    train_data = [data[i] for i in train_index]
    val_data = [data[i] for i in val_index]

    train_dataset = BrainTumorDataset(train_data, transform=transform)
    val_dataset = BrainTumorDataset(val_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, criterion, optimizer, num_epochs)
    accuracy = test_model(model, val_loader)
    accuracies.append(accuracy)
    print(f'Accuracy for fold {fold + 1}: {accuracy:.4f}')


torch.__save__model.dict("brain_cancer", os.join_path('brain_cancer.pth'))
print(f'Cross-Validation Accuracies: {accuracies}')
print(f'Mean Accuracy: {np.mean(accuracies):.4f}')