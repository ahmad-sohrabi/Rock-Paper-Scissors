from torch import nn


class RPSModel1(nn.Module):
    def __init__(self, num_channels):
        super(RPSModel1, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(32, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(64, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Flatten(),
            nn.Dropout(0.4),

            nn.Linear(51520, 512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 3)
        )

    def forward(self, x):
        return self.net(x)


class RPSModel2(nn.Module):
    def __init__(self, num_channels):
        super(RPSModel2, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(32, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(64, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(64, 128, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Flatten(),
            nn.Dropout(0.4),

            nn.Linear(20480, 512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 3)
        )

    def forward(self, x):
        return self.net(x)


class RPSModel3(nn.Module):
    def __init__(self, num_channels):
        super(RPSModel3, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(32, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(64, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(64, 128, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(128, 128, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(128, 256, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Flatten(),
            nn.Dropout(0.4),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 3)
        )

    def forward(self, x):
        return self.net(x)


class RPSModel4(nn.Module):
    def __init__(self, num_channels):
        super(RPSModel4, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(32, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(64, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(64, 128, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(128, 128, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Flatten(),
            nn.Dropout(0.4),

            nn.Linear(3584, 512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 3)
        )

    def forward(self, x):
        return self.net(x)
