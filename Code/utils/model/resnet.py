import torch
from torch import nn
from torch.nn import functional as F


class Resnet(nn.Module):

    def __init__(self, in_chans, out_chans, nker=384, learning_type="plain",
                norm="bnorm", nblk=16):
        super(Resnet, self).__init__()
        
        self.learning_type = learning_type

        # encoder part
        self.enc = CBR(in_channels, nker, kernel_size=3, stride=1,
                        padding=1, bias=True, norm=None, relu=0.0)
        
        # resblk part
        res = []
        for i in range(nblk):
            res += [ResBlock(nker, nker, kernel_size=3, stride=1,
                            padding=1, bias=True, norm=norm, relu=0.0)]
        self.res = nn.Sequential(* res)

        # decoder part
        self.dec = CBR(nker, nker, kernel_size=3, stride=1, padding=1,
                        bias=True, norm=norm, relu=0.0)
        
        # final convolution part
        self.fc = nn.Conv2d(in_channels=nker, out_channels=out_channels,
                            kernel_size=1, stride=1, padding=0, bias=True)

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

        x0 = x
        x = self.enc(x)
        x = self.res(x)
        x = self.dec(x)
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