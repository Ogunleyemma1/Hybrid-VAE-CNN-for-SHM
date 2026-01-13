# cnn_model.py
import torch
import torch.nn as nn

SEQ_LEN = 200
NUM_FEATURES = 4  # raw 4 channels

class CNN(nn.Module):
    """
    Input:  (B, 1, SEQ_LEN, NUM_FEATURES) = (B, 1, 200, 4)
    Output: (B, 2) logits for [SF, E]
    """
    def __init__(self, input_channels=1, num_classes=2, dropout_rate=0.4):
        super().__init__()

        def block(cin, cout, kt, kf, pt, pf):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=(kt, kf), padding=(pt, pf)),
                nn.GroupNorm(num_groups=8, num_channels=cout),
                nn.SiLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(input_channels, 32, 7, 3, 3, 1),
            nn.MaxPool2d(kernel_size=(2, 1)),  # 200 -> 100

            block(32, 64, 5, 3, 2, 1),
            nn.MaxPool2d(kernel_size=(2, 1)),  # 100 -> 50

            block(64, 128, 5, 3, 2, 1),
            nn.MaxPool2d(kernel_size=(2, 1)),  # 50 -> 25

            block(128, 256, 3, 3, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1)),      # (B,256,1,1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
