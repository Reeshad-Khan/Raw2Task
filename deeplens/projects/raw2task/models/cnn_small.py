import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, in_channels=1, width=64, num_classes=10):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, w, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(w),
            nn.Conv2d(w, w, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(w, 2*w, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(2*w),
            nn.Conv2d(2*w, 2*w, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2*w, 4*w, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.head = nn.Linear(4*w, num_classes)

    def forward(self, x):
        z = self.net(x).flatten(1)
        return self.head(z)
