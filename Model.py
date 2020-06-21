# Imports
from torch import nn


class ConvNet(nn.Module):
    """Convolutional Neural Network class"""

    def __init__(self):
        super(ConvNet, self).__init__()

        self.convLayer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.convLayer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.drop_out = nn.Dropout(p=0.5)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=(25 * 25 * 64), out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=2))

    def forward(self, x):
        """forward pass"""

        x = self.convLayer1(x)
        x = self.convLayer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)

        return x
