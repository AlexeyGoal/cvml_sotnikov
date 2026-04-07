import cv2
import torch
import torch.nn as nn
from pathlib import Path
from collections import deque
import torchvision
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = Path(__file__).parent / 'model.pth'

weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
model = torchvision.models.efficientnet_b0(weights)
for param in model.features.parameters():
    param.requires_grad = False
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
if model_path.exists():
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model = model.to(device)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
criterion = nn.BCEWithLogitsLoss()


transform = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Buffer:
    def __init__(self, maxsize=16):
        self.frames, self.labels = deque(maxlen=maxsize), deque(maxlen=maxsize)
    def append(self, frame, label):
        self.frames.append(transform(frame).to(device))
        self.labels.append(label)
    def __len__(self):
        return len(self.frames)
    def get_batch(self):
        return torch.stack(list(self.frames)), torch.tensor(list(self.labels), dtype=torch.float32).to(device)


def train(buffer):
    if len(buffer) < 10: return
    model.train()
    images, labels = buffer.get_batch()
    optimizer.zero_grad()
    loss = criterion(model(images).squeeze(1), labels)
    loss.backward()
    optimizer.step()
    return loss.item()

cap = cv2.VideoCapture(0)
buffer = Buffer()
count = 0

while True:
    _, frame = cap.read()
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'): break
    elif key == ord('1'): buffer.append(frame, 1.0); count += 1
    elif key == ord('2'): buffer.append(frame, 0.0); count += 1
    elif key == ord('s'): torch.save(model.state_dict(), model_path); print(f"Сохранено в {model_path}")
    
    if count >= buffer.frames.maxlen:
        loss = train(buffer)
        if loss: print(f'Loss = {loss:.4f}')
        count = 0

cap.release()
cv2.destroyAllWindows()