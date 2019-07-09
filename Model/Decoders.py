import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

class SegDecoder(nn.Module):  # based on PPM
    def __init__(self, n_class, n_encoded_channels,
                 scale_up=False, pool_scales=(2, 4, 8)):
        super(SegDecoder, self).__init__()
        self.scale_up = scale_up

        out_channels_1 = 64
        out_channels_2 = 64

        bias=True

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels=n_encoded_channels,
                          out_channels=out_channels_1,
                          kernel_size=1, bias=bias),
                nn.BatchNorm2d(out_channels_1),
                nn.ReLU(inplace=True)
            ))
        self.l1 = nn.ModuleList(self.ppm)

        self.l2 = nn.Sequential(
            nn.Conv2d(in_channels=n_encoded_channels + len(pool_scales) * out_channels_1,
                      out_channels=out_channels_2,
                      kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels_2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels_2, n_class, kernel_size=1)
        )


    def forward(self, encoded_features, segSize=None):
        '''
        :param encoded_features: first item is before fc with spatial integrity , second item is flattened conv features through fc
        :param segSize:
        :return:
        '''
        input_size = encoded_features.size()
        ppm_out = [encoded_features]
        for pool_scale in self.ppm:
            ppm_out.append(F.interpolate(
                pool_scale(encoded_features),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        x = torch.cat(ppm_out, 1)
        x = self.l2(x)
        if self.scale_up:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
        x = nn.functional.softmax(x, dim=1)
        return x

