from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):  # @save
    """The Residual block of ResNet."""

    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1, num_conv=2, batch_norm=True, use_relu=True):
        super().__init__()
        if num_conv == 1:
            self.conv1 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=3, padding=1, stride=strides)
            self.conv2 = None
        else:
            self.conv1 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=3, padding=1, stride=strides)
            self.conv2 = nn.Conv2d(num_channels, num_channels,
                                   kernel_size=3, padding=1)
        # if use_maxpool or only_maxpool:
        #     self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # else:
        #     self.max_pool = None

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(num_channels)
            if num_conv == 1:
                self.bn2 = None
            else:
                self.bn2 = nn.BatchNorm2d(num_channels)

        else:
            self.bn1 = None
            self.bn2 = None

        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, X):
        # if self.conv2 is None:
        #     Y = self.max_pool(self.conv1(X))
        #     # Y = self.conv1(X)
        #     return Y
        ## With BN and ReLU
        if self.bn1 and self.relu:
            Y = self.relu(self.bn1(self.conv1(X)))
            if self.conv2:
                Y = self.bn2(self.conv2(Y))

        ## With BN, no ReLU
        elif self.bn1:
            Y = self.bn1(self.conv1(X))
            if self.conv2:
                Y = self.bn2(self.conv2(Y))

        ## With ReLU, no BN
        elif self.relu:
            Y = self.relu(self.conv1(X))
            if self.conv2:
                Y = self.conv2(Y)

        ## No BN, no ReLU
        else:
            Y = self.conv1(X)
            if self.conv2:
                Y = self.conv2(Y)

        # if self.max_pool:
        #     Y = self.max_pool(Y)

        if self.conv3:
            X = self.conv3(X)

        Y += X

        ## With ReLU
        if self.relu:
            return self.relu(Y)

        ## No ReLU
        return Y

def resnet_block(input_channels, num_channels, num_residuals, num_conv=2,
                 first_block=False, batch_norm=True, use_relu=True):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2, num_conv=num_conv,
                                batch_norm=batch_norm, use_relu=use_relu))
        else:
            blk.append(Residual(num_channels, num_channels, num_conv=num_conv,
                                batch_norm=batch_norm, use_relu=use_relu))
    return blk