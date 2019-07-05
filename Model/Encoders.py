import torch
import torch.nn as nn
import torch.nn.functional as F


use_cpu = False
device = torch.device('cpu') if use_cpu else torch.device('cuda')
type = torch.float32 #if use_cpu else torch.float32 #xentropy doesn't support float16
args = {'device': device, 'dtype': type}



class SegEncoder(nn.Module):
    def __init__(self, in_shape, n_class=1):
        super(SegEncoder, self).__init__()

        bias = False
        channels = in_shape[1] if len(in_shape) > 3 else 1
        strideC_1 = 1

        kernel_sizeC_1 = 5
        out_channels_1 = 16

        self.l1 = nn.Sequential(
            # dw
            nn.Conv2d(in_channels=channels,
                      out_channels=out_channels_1, kernel_size=kernel_sizeC_1,
                      stride=strideC_1, padding=kernel_sizeC_1 // 2),
            nn.BatchNorm2d(out_channels_1),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(in_channels=out_channels_1,
                      out_channels=out_channels_1, kernel_size=kernel_sizeC_1,
                      stride=strideC_1, padding=kernel_sizeC_1 // 2),
            nn.BatchNorm2d(out_channels_1),
        )

        strideC_2 = 1
        kernel_sizeC_2 = 5
        out_channels_2 = 16
        self.l2 = nn.Sequential(
            # dw
            nn.Conv2d(in_channels=channels,
                      out_channels=out_channels_2, kernel_size=kernel_sizeC_2,
                      stride=strideC_2, padding=kernel_sizeC_2 // 2),
            nn.BatchNorm2d(out_channels_1),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(in_channels=out_channels_2,
                      out_channels=out_channels_2, kernel_size=kernel_sizeC_2,
                      stride=strideC_2, padding=kernel_sizeC_2 // 2),
            nn.BatchNorm2d(out_channels_2),
        )
        final_dim = (out_channels_2 // strideC_2) * out_channels_2
        self.fc = nn.Linear(in_features=final_dim, out_features=n_class)
        self.out_shape = n_class

    def forward(self, x):
        # Computes the activation of the first convolution, size will be size of input + padding - kernel size//2
        # get size of memory allocation in bytes : tensorname.element_size() * tensorname.nelement()
        l1 = self.l1(x)
        l2 = self.conv2(l1)
        l2_flattened = l2.view(l2.size(0), -1)
        fc = self.fc(l2_flattened)
        conv_out = [l2, fc]
        return conv_out
