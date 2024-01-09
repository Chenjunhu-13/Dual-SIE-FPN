import torch
from torch import nn

class SEBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels//r),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//r, in_channels),
            nn.Sigmoid(),
        )
        # self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)

        # Fscale
        y = torch.mul(x, y)
        return y


if __name__ == '__main__':
    in_tensor = torch.ones((1, 32, 4, 4))

    cb = SEBlock(in_channels=32, r=16)

    out_tensor = cb(in_tensor)

    print(in_tensor.shape)
    print(out_tensor.shape)