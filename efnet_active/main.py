import torch
import cv2
import torch.nn as nn
import torchvision
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
model = torchvision.models.efficientnet_b0(weights)
for param in model.features.parameters():
    param.requires_grad = False
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)


model.load_state_dict(torch.load('model.pth', map_location=device, weights_only=True))
model = model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    cv2.imshow("Camera", frame)
    
    if (cv2.waitKey(1) & 0xFF) == ord('p'):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor)).item()
        
        print('person' if prob > 0.5 else 'no person', f'{prob:.3f}')
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
