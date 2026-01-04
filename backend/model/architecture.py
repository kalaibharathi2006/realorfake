import torch
import torch.nn as nn
import torch.nn.functional as F

# A simple Mesonet-like architecture or efficient net for deepfake detection
# For this demo, we use a custom CNN
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeDetector, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 14 * 14, 128) # Assuming 224x224 input -> 14x14 feature map
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input shape: (Batch, 3, 224, 224)
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # -> 112
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # -> 56
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) # -> 28
        x = self.pool(F.relu(self.bn4(self.conv4(x)))) # -> 14
        
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_model(device='cpu'):
    model = DeepfakeDetector()
    model.to(device)
    return model
