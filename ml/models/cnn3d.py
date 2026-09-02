import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporal3DCNN(nn.Module):
    """
    Custom 3D Convolutional Neural Network (3D-CNN) for Spatio-Temporal Action & Gesture Recognition in NEXORA AI.
    Processes 16-frame video tensor inputs of shape (Batch, 3, 16, 112, 112).
    """
    def __init__(self, num_classes: int = 10, dropout_prob: float = 0.5):
        super(SpatioTemporal3DCNN, self).__init__()
        
        # Layer 1: 3D Conv -> BatchNorm -> ReLU -> MaxPool3d
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # Preserves frame temporal depth initially
        
        # Layer 2: 3D Conv -> BatchNorm -> ReLU -> MaxPool3d
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Layer 3: 3D Conv -> BatchNorm -> ReLU -> MaxPool3d
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Fully Connected Layers
        # Feature map dimensions at this point: 256 channels x 4 frames x 14 height x 14 width = 200,704
        self.fc1 = nn.Linear(256 * 4 * 14 * 14, 512)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 3D spatio-temporal architecture.
        Input shape: (B, 3, 16, 112, 112)
        Output shape: (B, num_classes)
        """
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten spatio-temporal features
        x = x.view(x.size(0), -1)
        
        # Classification Head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits

def get_3d_cnn_model(num_classes: int = 10) -> SpatioTemporal3DCNN:
    """Factory function returning initialized 3D-CNN instance."""
    return SpatioTemporal3DCNN(num_classes=num_classes)
