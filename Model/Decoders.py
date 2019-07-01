import torch
import torch.nn as nn
import torch.nn.functional as F

use_cpu = False
device = torch.device('cpu') if use_cpu else torch.device('cuda')
type = torch.float32  # if use_cpu else torch.float32 #xentropy doesn't support float16
args = {'device': device, 'dtype': type}


class SegDecoder(nn.Module):  # based on PPM
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(SegDecoder, self).__init__()
        self.use_softmax = use_softmax

        out_channels_1 = 256
        out_channels_2 = 256

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels=fc_dim,
                          out_channels=out_channels_1,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels_1),
                nn.ReLU(inplace=True)
            ))
        self.l1 = nn.ModuleList(self.ppm)

        self.l2 = nn.Sequential(
            nn.Conv2d(in_channels=fc_dim + len(pool_scales) * 512, #fix this dimenion, read github, understand it
                      out_channels=out_channels_2,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(F.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = F.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = F.softmax(x, dim=1)
            # class0 = out.data[0,:,0,0]
            # print(class0.shape)
            # print(torch.sum(class0))
        else:
            x = F.log_softmax(x, dim=1)

        return x
