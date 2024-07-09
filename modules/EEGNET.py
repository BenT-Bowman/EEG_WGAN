import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, in_channels:int= 1, filters: list = [8, 16, 16], kernel_size:int = 5,  dropout_rate=0.5, num_classes=1):
        super().__init__()
        pool_size, num_samples = 2, 500
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=filters[0], kernel_size=(1, kernel_size), padding=(0, kernel_size// 2), bias=False),
            nn.BatchNorm2d(filters[0], False),
            self._regularization(pool_size, dropout_rate),

            nn.Conv2d(filters[0], out_channels=filters[1], kernel_size=(in_channels, 1),groups=filters[0], bias=False),
            nn.BatchNorm2d(filters[1], False),
            self._regularization(pool_size, dropout_rate),

            nn.Conv2d(filters[1], out_channels= filters[2], kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(filters[2], False),
            self._regularization(pool_size, dropout_rate),

            nn.Conv2d(filters[2], out_channels= filters[2], kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(filters[2], False),
            self._regularization(pool_size, dropout_rate),

            nn.Flatten(),

            nn.Linear(9424, num_classes)
        )
        # print(filters[2] * ((num_samples // pool_size) // pool_size))
    def _regularization(self, pool_size, dropout_rate):
        return nn.Sequential(
            nn.ELU(),
            nn.AvgPool2d((1, pool_size)),
            nn.Dropout(dropout_rate),
            )

    def forward(self, x):
        return self.model(x)
    
class EEGNetClassifier(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = EEGNet(**kwargs)
    def forward(self, x):
        return F.sigmoid(self.model(x))
    
class EEGNetModified(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = EEGNet(**kwargs)
    def forward(self, x):
        out = x.view(x.size(0), 1, 19, 500)
        return self.model(out)


class EEGNet_SeqFirst(nn.Module):
    def __init__(self, in_channels:int= 1, filters: list = [8, 16, 16], kernel_size:int = 5,  dropout_rate=0.5, num_classes=1):
        super().__init__()
        pool_size, num_samples = 2, 150
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=filters[0], kernel_size=(kernel_size, 1), padding=(kernel_size// 2, 0), bias=False),
            nn.BatchNorm2d(filters[0], False),
            self._regularization(pool_size, dropout_rate),

            nn.Conv2d(filters[0], out_channels=filters[1], kernel_size=(1, in_channels),groups=filters[0], bias=False),
            nn.BatchNorm2d(filters[1], False),
            self._regularization(pool_size, dropout_rate),

            nn.Conv2d(filters[1], out_channels= filters[2], kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(filters[2], False),
            self._regularization(pool_size, dropout_rate),

            nn.Conv2d(filters[2], out_channels= filters[2], kernel_size=(16,1), padding=(8, 0), bias=False),
            nn.BatchNorm2d(filters[2], False),
            self._regularization(pool_size, dropout_rate),

            nn.Flatten(),

            nn.Linear(16*19*9, num_classes)
        )
        # print(filters[2] * ((num_samples // pool_size) // pool_size))
    def _regularization(self, pool_size, dropout_rate):
        return nn.Sequential(
            nn.ELU(),
            nn.AvgPool2d((pool_size, 1)),
            nn.Dropout(dropout_rate),
            )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    e = EEGNet_SeqFirst()
    noise = torch.randn(8, 1, 150, 19)
    print(e(noise).shape)

    e = EEGNet()
    noise = torch.randn(8, 1, 19, 500)
    print(e(noise).shape)