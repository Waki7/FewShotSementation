import torch.nn as nn


class SegEncoder(nn.Module): # will maintain same shape as input
    def __init__(self, in_shape, model_size=32, out_shape = 128,
                 dilation = 4, encoding_downsample=1):
        super(SegEncoder, self).__init__()

        def __init__(
                self, n_classes=21, block_config=[3, 4, 23, 3], input_size=(473, 473), version=None
        ):
            super(pspnet, self).__init__()

            self.block_config = (
                pspnet_specs[version]["block_config"] if version is not None else block_config
            )
            self.n_classes = pspnet_specs[version]["n_classes"] if version is not None else n_classes
            self.input_size = pspnet_specs[version]["input_size"] if version is not None else input_size

            # Encoder
            self.convbnrelu1_1 = conv2DBatchNormRelu(
                in_channels=3, k_size=3, n_filters=64, padding=1, stride=2, bias=False
            )
            self.convbnrelu1_2 = conv2DBatchNormRelu(
                in_channels=64, k_size=3, n_filters=64, padding=1, stride=1, bias=False
            )
            self.convbnrelu1_3 = conv2DBatchNormRelu(
                in_channels=64, k_size=3, n_filters=128, padding=1, stride=1, bias=False
            )

            # Vanilla Residual Blocks
            self.res_block2 = residualBlockPSP(self.block_config[0], 128, 64, 256, 1, 1)
            self.res_block3 = residualBlockPSP(self.block_config[1], 256, 128, 512, 2, 1)

            # Dilated Residual Blocks
            self.res_block4 = residualBlockPSP(self.block_config[2], 512, 256, 1024, 1, 2)
            self.res_block5 = residualBlockPSP(self.block_config[3], 1024, 512, 2048, 1, 4)

    def calc_padding(self, kernel, dilation):
        return (dilation)*(kernel//2)

    def forward(self, x):
        # Computes the activation of the first convolution, size will be size of input + padding - kernel size//2
        # get size of memory allocation in bytes : tensorname.element_size() * tensorname.nelement()
        l1 = self.l1(x)
        l2 = self.l2(l1)
        # l2_flattened = l2.view(l2.size(0), -1)
        # fc = self.fc(l2_flattened)
        return l2

