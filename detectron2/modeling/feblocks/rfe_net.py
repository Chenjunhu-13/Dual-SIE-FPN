import torch
from torch import nn


class RFEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RFEBlock, self).__init__()
        self.conv1x1_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv3x3_1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3x3_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv3x3_3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu4 = nn.ReLU(inplace=True)

        # self.w = nn.Parameter(torch.ones(2))

    def forward(self, x):
        # w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        # w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        out1 = self.relu1(self.bn1(self.conv1x1_1(x)))

        out2 = self.relu2(self.bn2(self.conv3x3_1(out1)))
        out2 = self.relu3(self.bn3(self.conv3x3_2(out2)))
        out2 = self.relu4(self.bn4(self.conv3x3_3(out2)))

        # print(out2.shape)
        # out = w1 * out1 + w2 * out2
        # out = out1 + out2
        out = out1 + out2

        return out


if __name__ == '__main__':
    in_tensor = torch.ones((1, 256, 4, 4))

    cb = RFEBlock(in_channels=256, out_channels=256)

    out_tensor = cb(in_tensor)

    print(in_tensor.shape)
    print(out_tensor.shape)