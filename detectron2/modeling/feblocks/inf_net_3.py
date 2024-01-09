import torch
import torch.nn.functional as F
from torch import nn
import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, ShapeSpec, get_norm
class INFBlock(nn.Module):
    def __init__(self, in_c, norm=""):
        super(INFBlock, self).__init__()
        
        self.w1 = nn.Parameter(torch.ones(3))
        self.w2 = nn.Parameter(torch.ones(3))
        self.w3 = nn.Parameter(torch.ones(3))

        # self.w1 = nn.Parameter(torch.ones(4))
        # self.w2 = nn.Parameter(torch.ones(4))
        # self.w3 = nn.Parameter(torch.ones(4))
        # self.w4 = nn.Parameter(torch.ones(4))

    def forward(self, x):
        inf_features = []
        w1_1 = torch.exp(self.w1[0]) / torch.sum(torch.exp(self.w1))
        w1_2 = torch.exp(self.w1[1]) / torch.sum(torch.exp(self.w1))
        w1_3 = torch.exp(self.w1[2]) / torch.sum(torch.exp(self.w1))
        # w1_4 = torch.exp(self.w1[3]) / torch.sum(torch.exp(self.w1))
        #
        # x[0] = w1_1 * x[0] + w1_2 * F.interpolate(x[1], scale_factor=2.0, mode="bilinear") + \
        #              w1_3 * F.interpolate(x[2], scale_factor=4.0, mode="bilinear") + \
        #              w1_4 * F.interpolate(x[3], scale_factor=8.0, mode="bilinear")

        x[0] = w1_1 * x[0] + w1_2 * F.interpolate(x[1], scale_factor=2.0, mode="bilinear") + \
               w1_3 * F.interpolate(x[2], scale_factor=4.0, mode="bilinear")

        inf_features.append(x[0])

        w2_1 = torch.exp(self.w2[0]) / torch.sum(torch.exp(self.w2))
        w2_2 = torch.exp(self.w2[1]) / torch.sum(torch.exp(self.w2))
        w2_3 = torch.exp(self.w2[2]) / torch.sum(torch.exp(self.w2))
        # w2_4 = torch.exp(self.w2[3]) / torch.sum(torch.exp(self.w2))
        #
        # x[1] = w2_1 * F.interpolate(x[0], scale_factor=1 / 2, mode="bilinear") + w2_2 * x[1] + \
        #              w2_3 * F.interpolate(x[2], scale_factor=2.0, mode="bilinear") + \
        #              w2_4 * F.interpolate(x[3], scale_factor=4.0, mode="bilinear")

        x[1] = w2_1 * F.interpolate(x[0], scale_factor=1 / 2, mode="bilinear") + w2_2 * x[1] + \
               w2_3 * F.interpolate(x[2], scale_factor=2.0, mode="bilinear")

        inf_features.append(x[1])

        w3_1 = torch.exp(self.w3[0]) / torch.sum(torch.exp(self.w3))
        w3_2 = torch.exp(self.w3[1]) / torch.sum(torch.exp(self.w3))
        w3_3 = torch.exp(self.w3[2]) / torch.sum(torch.exp(self.w3))
        # w3_4 = torch.exp(self.w3[3]) / torch.sum(torch.exp(self.w3))
        #
        # x[2] = w3_1 * F.interpolate(x[0], scale_factor=1 / 4, mode="bilinear") + \
        #              w3_2 * F.interpolate(x[1], scale_factor=1 / 2, mode="bilinear") + w3_3 * x[2] + \
        #              w3_4 * F.interpolate(x[3], scale_factor=2, mode="bilinear")

        x[2] = w3_1 * F.interpolate(x[0], scale_factor=1 / 4, mode="bilinear") + \
               w3_2 * F.interpolate(x[1], scale_factor=1 / 2, mode="bilinear") + w3_3 * x[2]

        inf_features.append(x[2])

        # w4_1 = torch.exp(self.w4[0]) / torch.sum(torch.exp(self.w4))
        # w4_2 = torch.exp(self.w4[1]) / torch.sum(torch.exp(self.w4))
        # w4_3 = torch.exp(self.w4[2]) / torch.sum(torch.exp(self.w4))
        # w4_4 = torch.exp(self.w4[3]) / torch.sum(torch.exp(self.w4))
        #
        # x[3] = w4_1 * F.interpolate(x[0], scale_factor=1 / 8, mode="bilinear") + \
        #              w4_2 * F.interpolate(x[1], scale_factor=1 / 4, mode="bilinear") + \
        #              w4_3 * F.interpolate(x[2], scale_factor=1 / 2, mode="bilinear") + \
        #              w4_4 * x[3]
        #
        # inf_features.append(x[3])

        return inf_features

