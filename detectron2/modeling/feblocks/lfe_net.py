import torch
from torch import nn
from .se_net import SEBlock
import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, ShapeSpec, get_norm

class RESBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RESBlock, self).__init__()
        self.conv1x1_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv3x3_1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3x3_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)



    def forward(self, x):
        # w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        # w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        out1 = self.relu1(self.bn1(self.conv1x1_1(x)))
        out2 = self.relu2(self.bn2(self.conv3x3_1(out1)))
        out2 = self.relu3(self.bn3(self.conv3x3_2(out2)))
        # out = w1 * out1 + w2 * out2
        out = out1 + out2
        return out


class LFEBlock(nn.Module):
    def __init__(self, in_channels, norm=""):
        super(LFEBlock, self).__init__()
        # use_bias = norm == ""
        # norm1 = get_norm(norm, in_channels)
        # norm2 = get_norm(norm, in_channels)
        # norm3 = get_norm(norm, in_channels)
        # norm4 = get_norm(norm, in_channels)
        # norm5 = get_norm(norm, in_channels)

        self.conv3x3_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv1x1_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        # self.bn2 = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()

        self.conv3x3_2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

        self.conv3x3_3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

        self.conv3x3_4 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

        self.seblock = SEBlock(in_channels=in_channels, r=16)

        self.w = nn.Parameter(torch.ones(3))

        # self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        # self.w_relu = nn.ReLU()
        # self.e = 1e-4

        # self.resblock = RESBlock(in_channels=in_channels, out_channels=in_channels)
        # weight_init.c2_xavier_fill(self.conv3x3_1)
        # weight_init.c2_xavier_fill(self.conv1x1_1)
        # weight_init.c2_xavier_fill(self.conv3x3_2)
        # weight_init.c2_xavier_fill(self.conv3x3_3)
        # weight_init.c2_xavier_fill(self.conv3x3_4)

    def forward(self, x):
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        # print(self.w)
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        # w_lfe = self.w_relu(self.w)
        # w = w_lfe / (torch.sum(w_lfe) + self.e)
        # w1, w2, w3 = w[0], w[1], w[2]

        y1 = self.relu_1(self.conv3x3_1(x))
        y1 = self.sigmoid(self.conv1x1_1(y1))

        # y1 = self.relu_1(self.bn1(self.conv3x3_1(x)))
        # y1 = self.sigmoid(self.bn2(self.conv1x1_1(y1)))

        y2 = self.conv3x3_2(x)
        z1 = self.conv3x3_3(torch.mul(y1, y2))

        z2 = self.conv3x3_4(x)
        out1 = z1 + z2

        out2 = self.seblock(x)

        out = w1 * out1 + w2 * out2 + w3 * x
        # out_res = self.resblock(x)

        # out = out + out_res
        # out = out1 + out2 + x
        # out = self.conv3x3_5(out)

        return out


if __name__ == '__main__':
    in_tensor = torch.ones((1, 32, 4, 4))

    cb = LFEBlock(in_channels=32)

    out_tensor = cb(in_tensor)

    print(in_tensor.shape)
    print(out_tensor.shape)