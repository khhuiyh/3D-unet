import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tensorboardX import SummaryWriter


class convblock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(convblock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                                stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)

    def forward(self, x):
        out = self.batch_norm(self.conv3d(x))
        # out = self.conv3d(out)
        out = F.elu(out, inplace=True)
        return out


class conv_layers_encoder(nn.Module):
    def __init__(self, in_channels, num_conv_blocks=2, depth=0):
        super(conv_layers_encoder, self).__init__()
        self.in_channels = in_channels
        self.num_conv_blocks = num_conv_blocks
        self.conv_block = nn.ModuleList([])
        self.depth = depth
        if self.depth == 0:
            out_channels = 32
        else:
            out_channels = in_channels
        for i in range(num_conv_blocks):
            self.conv_block.append(convblock(in_channels=in_channels, out_channels=out_channels))
            in_channels = out_channels
            out_channels = out_channels * 2

    def forward(self, x):
        for i in self.conv_block:
            x = i(x)
        return x


class conv_layers_decoder(nn.Module):
    def __init__(self, out_channels, num_conv_blocks=2):
        super(conv_layers_decoder, self).__init__()
        self.num_conv_blocks = num_conv_blocks
        self.in_channels = out_channels * (2 ** (self.num_conv_blocks - 1))
        self.conv_block = nn.ModuleList([])
        out_channels = self.in_channels // 2
        for i in range(num_conv_blocks):
            if i == num_conv_blocks - 1:
                out_channels = self.in_channels
                self.conv_block.append(convblock(in_channels=self.in_channels, out_channels=out_channels))
            elif i == 0:
                self.conv_block.append(convblock(
                    in_channels=self.in_channels + self.in_channels // (2 ** (num_conv_blocks - 1)),
                    out_channels=out_channels))
                self.in_channels = out_channels
                out_channels = out_channels // 2
            else:
                self.conv_block.append(convblock(in_channels=self.in_channels, out_channels=out_channels))
                self.in_channels = out_channels
                out_channels = out_channels // 2

    def forward(self, x):
        for i in self.conv_block:
            x = i(x)
        return x


class thrdunet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, model_depth=4, pool_size=2, num_conv_blocks=2, dev='gpu'):
        super(thrdunet, self).__init__()
        self.mylist = []
        self.dev = dev
        self.num_conv_blocks = num_conv_blocks
        self.model_depth = model_depth
        self.conv3d = nn.Conv3d(in_channels=32 * (2 ** (num_conv_blocks - 1)), out_channels=out_channels, kernel_size=3,
                                stride=1, padding=1)
        self.conv_block_encoder = nn.ModuleList([])
        self.conv_block_decoder = nn.ModuleList([])
        self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=2, padding=0)
        self.encoder_list = []
        root = 64 * (2 ** (num_conv_blocks - 2))
        for depth in range(model_depth):
            if depth == 0:
                intch = in_channels
            else:
                intch = root * ((2 ** (num_conv_blocks - 1)) ** (depth - 1))
            self.conv_block_encoder.append(conv_layers_encoder(in_channels=intch,
                                                               num_conv_blocks=num_conv_blocks,
                                                               depth=depth))
        for depth in range(model_depth - 2, -1, -1):
            outch = root * ((2 ** (num_conv_blocks - 1)) ** depth)
            self.conv_block_decoder.append(conv_layers_decoder(out_channels=outch,
                                                               num_conv_blocks=num_conv_blocks))

    def forward(self, x):
        self.mylist.clear()
        self.encoder_list.clear()
        for i in self.conv_block_encoder:
            x = i(x)
            if not (i == self.conv_block_encoder[-1]):
                self.encoder_list.append(x)
                self.mylist.append(tuple(x.shape)[2:5])
                x = self.pooling(x)
        for i in self.conv_block_decoder:
            x = F.interpolate(x, self.mylist[-1])
            x = torch.cat((x, self.encoder_list[-1]), dim=1)
            self.mylist.pop()
            self.encoder_list.pop()
            x = i(x)
        x = self.conv3d(x)
        x = torch.sigmoid(x)
        return x


class upsample_block(nn.Module):
    def __init__(self, out_channels, shape):
        super(upsample_block, self).__init__()
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)
        self.shape = shape

    def forward(self, x):
        out = self.batch_norm(F.interpolate(x, self.shape, mode='trilinear', align_corners=True))
        out = F.elu(out)
        return out


class upsample_block_transpose(nn.Module):
    def __init__(self, out_channels):
        super(upsample_block_transpose, self).__init__()
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)
        self.up_fun = nn.ConvTranspose3d(out_channels, out_channels, 3, 2, 1, 1)

    def forward(self, x):
        out = self.batch_norm(self.up_fun(x))
        out = F.elu(out)
        return out


class mythrdunet(nn.Module):
    def __init__(self, in_channels=1, pool_size=2):
        super(mythrdunet, self).__init__()
        self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=2, padding=0)
        self.layer1 = nn.Sequential(convblock(in_channels, 8), convblock(8, 16))
        self.layer2 = nn.Sequential(convblock(16, 16), convblock(16, 32))
        self.layer3 = nn.Sequential(convblock(32, 32), convblock(32, 64))
        self.layer4 = nn.Sequential(convblock(64, 64), convblock(64, 128))
        self.layer5 = nn.Sequential(convblock(128 + 64, 64), convblock(64, 64))
        self.layer6 = nn.Sequential(convblock(32 + 64, 32), convblock(32, 32))
        self.conv3d = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.up1 = upsample_block(128, (16, 16, 16))
        self.up2 = upsample_block(64, (32, 32, 32))
        self.up3 = upsample_block(32, (64, 64, 64))

    def forward(self, x):
        # encode
        x1 = self.layer1(x)
        x2 = self.layer2(self.pooling(x1))
        x3 = self.layer3(self.pooling(x2))

        out = self.layer4(self.pooling(x3))
        # decode
        out = torch.cat((self.up1(out), x3), dim=1)
        out = self.layer5(out)

        out = torch.cat((self.up2(out), x2), dim=1)
        out = self.layer6(out)
        out = self.up3(out)
        out = self.conv3d(out)
        return torch.sigmoid(out)


class mythrdunet_transpose(nn.Module):
    def __init__(self, in_channels=1, pool_size=2):
        super(mythrdunet_transpose, self).__init__()
        self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=2, padding=0)
        self.layer1 = nn.Sequential(convblock(in_channels, 8), convblock(8, 16))
        self.layer2 = nn.Sequential(convblock(16, 16), convblock(16, 32))
        self.layer3 = nn.Sequential(convblock(32, 32), convblock(32, 64))
        self.layer4 = nn.Sequential(convblock(64, 64), convblock(64, 128))
        self.layer5 = nn.Sequential(convblock(128 + 64, 64), convblock(64, 64))
        self.layer6 = nn.Sequential(convblock(32 + 64, 32), convblock(32, 32))
        self.conv3d = nn.Conv3d(in_channels=32 + 16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.up1 = upsample_block_transpose(128)
        self.up2 = upsample_block_transpose(64)
        self.up3 = upsample_block_transpose(32)

    def forward(self, x):
        # encode
        x1 = self.layer1(x)
        x2 = self.layer2(self.pooling(x1))
        x3 = self.layer3(self.pooling(x2))

        out = self.layer4(self.pooling(x3))
        # decode
        out = torch.cat((self.up1(out), x3), dim=1)
        out = self.layer5(out)

        out = torch.cat((self.up2(out), x2), dim=1)
        out = self.layer6(out)
        out = torch.cat((self.up3(out), x1), dim=1)
        out = self.conv3d(out)
        return torch.sigmoid(out)


if __name__ == "__main__":
    model = thrdunet(in_channels=1, out_channels=1, num_conv_blocks=2, model_depth=3, dev='cpu')
    x = torch.rand(1, 1, 64, 64, 64)
    x = model.forward(x)
