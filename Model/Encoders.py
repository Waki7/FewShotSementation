import torch
import torch.nn as nn
import torch.nn.functional as F


use_cpu = True
device = torch.device('cpu') if use_cpu else torch.device('cuda')
dtype = torch.float32 #if use_cpu else torch.float32 #xentropy doesn't support float16
args = {'device': device, 'dtype': dtype}



class SegEncoder(nn.Module): # will maintain same shape as input
    def __init__(self, in_shape, out_shape=32, dilation = 4):
        super(SegEncoder, self).__init__()

        bias = False
        channels = in_shape[1] if len(in_shape) > 3 else 1
        stride1 = 1
        kernel1 = 5

        out_channels_1 = 32

        self.l1 = nn.Sequential(
            # dw
            nn.Conv2d(in_channels=channels,
                      out_channels=out_channels_1, kernel_size=kernel1,
                      stride=stride1, padding=self.calcPadding(kernel1, dilation),
                      dilation=dilation),
            nn.BatchNorm2d(out_channels_1),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(in_channels=out_channels_1,
                      out_channels=out_channels_1, kernel_size=kernel1,
                      stride=stride1, padding=self.calcPadding(kernel1, dilation),
                      dilation=dilation),
            nn.BatchNorm2d(out_channels_1),
        )

        stride2 = 1
        kernel2 = 3
        out_channels_2 = 32
        self.l2 = nn.Sequential(
            # dw
            nn.Conv2d(in_channels=out_channels_1,
                      out_channels=out_channels_2, kernel_size=kernel2,
                      stride=stride2, padding=self.calcPadding(kernel2, dilation),
                      dilation=dilation),
            nn.BatchNorm2d(out_channels_1),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(in_channels=out_channels_2,
                      out_channels=out_shape, kernel_size=kernel2,
                      stride=stride2, padding=self.calcPadding(kernel2, dilation),
                      dilation=dilation),
            nn.BatchNorm2d(out_channels_2),
        )
        # final_dim = ((in_shape[-1]*in_shape[-2])//stride2) * out_channels_2
        # self.fc = nn.Linear(in_features=final_dim, out_features=n_class)
        self.out_shape = out_channels_2

    def calcPadding(self, kernel, dilation):
        return (dilation)*(kernel//2)

    def forward(self, x):
        # Computes the activation of the first convolution, size will be size of input + padding - kernel size//2
        # get size of memory allocation in bytes : tensorname.element_size() * tensorname.nelement()
        l1 = self.l1(x)
        l2 = self.l2(l1)
        # l2_flattened = l2.view(l2.size(0), -1)
        # fc = self.fc(l2_flattened)
        return l2

