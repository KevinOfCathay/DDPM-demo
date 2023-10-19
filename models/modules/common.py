import torch.nn as nn
import torch


class Concat(nn.Module):
    def __init__(self) -> None:
        super(Concat, self).__init__()

    def forward(self, x, y):
        return torch.concat([x, y], dim=1)


class AvgPool2x(nn.Module):
    def __init__(self):
        super(AvgPool2x, self).__init__()
        self.pool = nn.AvgPool2d(2, 2, 0)

    def forward(self, x):
        return self.pool(x)


class Conv3x3(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, strides: int = 1):
        '''
        ch_input: 输入通道数
        ch_output: 输出通道数
        '''
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor):
        return self.silu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        '''
        ch_input: 输入通道数
        ch_output: 输出通道数
        '''
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor):
        return self.silu(self.bn(self.conv(x)))


class Upsample2x(nn.Module):
    '''
    将图像的宽和高×2
    '''

    def __init__(self,
                 ch_in: int,
                 ch_out: int,
                 conv_t: bool = True) -> None:
        super(Upsample2x, self).__init__()

        # 选择使用 conv transpose 还是 nearest interpolation
        self.convt = conv_t

        if conv_t:
            self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1, bias=False)
        else:
            self.conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=1, padding=0)
        self.norm = nn.BatchNorm2d(ch_out)
        self.silu = nn.SiLU()

    def forward(self, x):
        if self.convt:
            upsample = self.up(x)
        else:
            upsample = self.conv(nn.functional.interpolate(x, scale_factor=2, mode="nearest"))
        return self.silu(self.norm(upsample))