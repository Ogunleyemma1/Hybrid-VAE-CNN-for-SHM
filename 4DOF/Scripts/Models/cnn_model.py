import torch
import torch.nn as nn

SEQ_LEN = 100
NUM_FEATURES = 12


class CNN(nn.Module):
    """
    2D Convolutional Neural Network for sequence-based fault classification.
    Input: (batch, 2, SEQ_LEN, NUM_FEATURES), e.g., (batch, 2, 100, 12)
    Output: (batch, 2) logits for binary classification.
    """
    def __init__(self, input_channels: int = 2, num_classes: int = 2, dropout_rate: float = 0.5):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 25 * 3, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.fc2 = nn.Linear(128, num_classes)

        # IMPORTANT: This must match old behavior
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# Compatibility alias so your newer scripts can still import CNNClassifier if needed
class CNNClassifier(CNN):
    def __init__(self, dropout_rate: float = 0.5):
        super().__init__(input_channels=2, num_classes=2, dropout_rate=dropout_rate)
