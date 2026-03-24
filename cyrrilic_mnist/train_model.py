import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset
import time
from sklearn.model_selection import train_test_split
from PIL import Image

class CyrrilicDataset(Dataset):
    
    def __init__(self, data_path, image_paths, class_to_idx, transform=None):
        
        self.data_path = Path(data_path)
        self.image_paths = image_paths
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
      
    def __getitem__(self, idx):
       
        path = self.image_paths[idx]
        
        
        img_path = self.data_path / path
        
        
        with Image.open(img_path) as img:
            
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            image = img.getchannel('A')
            image = np.array(image)
        
        
        image = np.expand_dims(image, axis=-1)
        
        
        label = self.class_to_idx[path.split('/')[0]]
        
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CyrillicCNN(nn.Module):
    def __init__(self):
        super(CyrillicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2) 
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 3 * 3, 256) 
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 34)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def get_data_info(data_path):
    
    data_path = Path(data_path)
    
    
    all_paths = []
    classes = []
    
    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir():
            class_name = class_dir.name
            classes.append(class_name)
            
            for img_path in class_dir.glob('*.png'):
                
                relative_path = f"{class_name}/{img_path.name}"
                all_paths.append(relative_path)
    
    
    class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
    
    
    
    return all_paths, class_to_idx


if __name__ == "__main__":
    
    path = Path(__file__).parent
    data_path = path / "Cyrillic"  
    model_path = path / "model_test.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")
    
    
    if not data_path.exists():
        print(f"Error: {data_path} is not found")
        
        exit(1)
    
    
    all_paths, class_to_idx = get_data_info(data_path)
    
    
    all_labels = [class_to_idx[f.split('/')[0]] for f in all_paths]
    
   
    train_paths, test_paths, train_labels, _ = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    print(f"Train samples: {len(train_paths)}")
    print(f"Test samples: {len(test_paths)}")
    
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    
    train_dataset = CyrrilicDataset(data_path, train_paths, class_to_idx, train_transform)
    test_dataset = CyrrilicDataset(data_path, test_paths, class_to_idx, test_transform)
    
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    
    model = CyrillicCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)
    
    train_loss_hist = []
    train_acc_hist = []
    
    
    if not model_path.exists():
        
        for epoch in range(15):
            start_time = time.perf_counter()
            model.train()
            run_loss, correct, total = 0.0, 0, 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                run_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            scheduler.step()
            epoch_loss = run_loss / len(train_loader)
            epoch_acc = 100 * (correct / total)
            train_loss_hist.append(epoch_loss)
            train_acc_hist.append(epoch_acc)
            
            t = time.perf_counter() - start_time
            print(f"Epoch: {epoch+1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Time: {t:.2f}s")
        
        
        torch.save(model.state_dict(), model_path)
        
        
        
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.title("Loss")
        plt.plot(train_loss_hist)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        
        plt.subplot(122)
        plt.title("Accuracy")
        plt.plot(train_acc_hist)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(path / "train.png")
        plt.show()
        
    else:
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        
    
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
