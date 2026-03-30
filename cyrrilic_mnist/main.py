from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from train_model import CyrillicCNN, CyrrilicDataset, get_data_info


    
path = Path(__file__).parent
data_path = path / 'Cyrillic'  
model_path = path / "model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not data_path.exists():
    print("Error: Cyrillic is not found")
    exit(1)  

if not model_path.exists():
    print(f"Error: Model is not found")
    exit(1)  

paths, class_mapping = get_data_info(data_path)
idx_to_class = {v: k for k, v in class_mapping.items()}
all_labels = [class_mapping[p.split('/')[0]] for p in paths]

from sklearn.model_selection import train_test_split
_, test_paths = train_test_split(
    paths, 
    test_size=0.2, 
    random_state=42, 
    stratify=all_labels
)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((28, 28)),
    transforms.Normalize((0.5,), (0.5,))
])


test_dataset = CyrrilicDataset(data_path, test_paths, class_mapping, test_transform)
test_loader = DataLoader(
    test_dataset, 
    batch_size=64, 
    shuffle=False,  
    num_workers=0
)


model = CyrillicCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

correct = 0
total = 0
class_correct = {}
class_total = {}
errors = []

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        preds = outputs.argmax(1)
        
    
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        
        for i in range(len(labels)):
            true_label = labels[i].item()
            pred_label = preds[i].item()
            
            class_total[true_label] = class_total.get(true_label, 0) + 1
            if true_label == pred_label:
                class_correct[true_label] = class_correct.get(true_label, 0) + 1
            else:
                errors.append({
                    'true': true_label,
                    'pred': pred_label,
                    'true_name': idx_to_class[true_label],
                    'pred_name': idx_to_class[pred_label]
                })
        
        if (batch_idx + 1) % 10 == 0:
            print(f"   Обработано: {batch_idx + 1}/{len(test_loader)} батчей")

accuracy = 100.0 * correct / total


print(f"Accuracy: {accuracy:.2f}%")

