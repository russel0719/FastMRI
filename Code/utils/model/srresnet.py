import torch
from torch import nn
from torch.nn import functional as F


class SRResnet(nn.Module):

    def __init__(self, in_chans, out_chans, nker, learning_type="plain",
                norm="bnorm", nblk=16):
        super(SRResnet, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.learning_type = learning_type
        
        # encoder part
        self.enc = CBR(in_channels, nker, kernel_size=9, stride=1,
                        padding=4, bias=True, norm=None, relu=0.0)
        
        # resblk part
        res = []
        for i in range(nblk):
            res += [ResBlock(nker, nker, kernel_size=3, stride=1,
                            padding=1, bias=True, norm=norm, relu=0.0)]
        self.res = nn.Sequential(* res)

        # decoder part
        self.dec = CBR(nker, nker, kernel_size=3, stride=1, padding=1,
                        bias=True, norm=norm, relu=None)
        
        # pixelshuffler part
        ps1 = []
        ps1 += [nn.Conv2d(in_channels=nker, out_channels=nker,
                        kernel_size=3, stride=1, padding=1)]
        ps1 += [PixelShuffle(ry=2, rx=2)]
        ps1 += [nn.ReLU()]
        self.ps1 = nn.Sequential(* ps1)
        ps2 = []
        ps2 += [nn.Conv2d(in_channels=nker, out_channels=nker,
                        kernel_size=3, stride=1, padding=1)]
        ps2 += [PixelShuffle(ry=2, rx=2)]
        ps2 += [nn.ReLU()]
        self.ps2 = nn.Sequential(* ps2)

        # final convolution part
        self.fc = nn.Conv2d(in_channels=nker, out_channels=out_channels,
                            kernel_size=9, stride=1, padding=4)

    def norm(self, x):
        b, h, w = x.shape
        x = x.view(b, h * w)
        mean = x.mean(dim=1).view(b, 1, 1)
        std = x.std(dim=1).view(b, 1, 1)
        x = x.view(b, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, x):
        x, mean, std = self.norm(x)
        x = x.unsqueeze(1)

        x = self.enc(x)
        x0 = x

        x = self.res(x)

        x = self.dec(x)
        x = x0 + x

        x = self.ps1(x)
        x = self.ps2(x)

        x = self.fc(x)

        if self.learning_type == "plain":
            x = self.fc(x)
        elif self.learning_type == "residual":
            x = x0 + self.fc(x)

        x = x.squeeze(1)
        x = self.unnorm(x, mean, std)

        return x


class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                stride=1, padding=1, bias=True, norm="bnorm",
                relu=0.0):
        super().__init__()

        layers = []
        # Convolution layer
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride,
                            padding=padding, bias=bias)]
        # Batch Normalization layer
        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(num_features=out_channels)]
        # ReLU layer
        if not relu is None:
            layers += [nn.ReLU() if relu == 0.0 else nn.LeakyReLU()]

        self.cbr = nn.Sequential(* layers)

    def forward(self, x):
        return self.cbr(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                stride=1, padding=1, bias=True, norm="bnorm",
                relu=0.0):
        super().__init__()

        layers = []
        # 1st CBR
        layers += [CBR(in_channels, out_channels, kernel_size=kernel_size,
                        stride=stride, padding=padding, bias=bias,
                        norm=norm, relu=relu)]
        # 2nd CBR
        layers += [CBR(in_channels, out_channels, kernel_size=kernel_size,
                        stride=stride, padding=padding, bias=bias,
                        norm=norm, relu=None)]
        
        self.resblk = nn.Sequential(* layers)

    def forward(self, x):
        return x + self.resblk

class PixelShuffle(nn.Module):
    def __init__(self, ry=2, rx=2):
        super().__init__()
        self.ry = ry
        self.rx = rx

    def forward(self, x):
        ry = self.ry
        rx = self.rx

        [B, C, H, W] = list(x.shape)

        x = x.reshape(B, C, H // ry, ry, W // rx, rx)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C * ry * rx, H // ry, W // rx)

        return x

class PixelUnshuffle(nn.Module):
    def __init__(self, ry=2, rx=2):
        super().__init__()
        self.ry = ry
        self.rx = rx

    def forward(self, x):
        ry = self.ry
        rx = self.rx

        [B, C, H, W] = list(x.shape)

        x = x.reshape(B, C // (ry * rx), ry, rx, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C // (ry * rx), H * ry, W * rx)

        return x