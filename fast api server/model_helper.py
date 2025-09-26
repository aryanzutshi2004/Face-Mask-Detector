import torch
import torch.nn as nn
from torchvision.models import resnet50
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
classes = ["With Mask", "Without Mask"]
num_classes = len(classes)

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet50(weights="DEFAULT")
        
        for params in self.model.parameters():
            params.requires_grad = False
        
        for params in self.model.layer4.parameters():
            params.requires_grad =True
        in_features = self.model.fc.in_features

        self.model.fc = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.model(x)
with torch.no_grad():  
    model = Model(num_classes).to(device)
    model.load_state_dict(torch.load("./face_mask_model.pth", map_location=device))

def give_prediction(image):
    with torch.no_grad():
        model.eval()
        img = transform(image).unsqueeze(0).to(device)
        prediction = model(img)
        _, predicted_label = torch.max(prediction, 1)
        return classes[predicted_label]