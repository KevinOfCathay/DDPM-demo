import torch.nn as nn
import torch
from models.modules.common import Conv3x3, Conv1x1, AvgPool2x, Upsample2x
from models.modules.attention import Attention


class TimestepBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ts_dims: int):
        '''
        channels: 输入/输出通道数
        '''
        super(TimestepBlock, self).__init__()

        self.ts_linear = nn.Linear(ts_dims, out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU()

    def forward(self, x: torch.Tensor, ts: torch.Tensor):
        '''
        x: 输入图片
        t: timestep
        '''
        x_conv_1 = self.conv1(x)
        t_linear_1 = self.ts_linear(ts)[:, :, None, None]

        x_add_t = torch.add(x_conv_1, t_linear_1)

        x_bn_1 = self.bn1(x_add_t)
        x_silu_1 = self.act1(x_bn_1)

        x_conv_2 = self.conv2(x_silu_1)
        x_bn_2 = self.bn2(x_conv_2)
        x_silu_2 = self.act2(x_bn_2)

        return x_silu_2


class ResAttentionModule(nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int, ts_dims: int, layers: int = 1,
                 attention: bool = True, downscale: bool = False, upsacle: bool = False) -> None:
        '''
        ### Args:
            - layers: 复制多少层, layer = 2 意味着层数翻倍
            - attention: 是否要加入 attention
            - downscale: 是否在最后加入一个 Average Pool
        '''
        super(ResAttentionModule, self).__init__()

        self.downscale = downscale
        self.upscale = upsacle
        self.attention = attention
        self.skip = (in_channels != out_channels)

        ts_blocks = []
        # 将 block 加入到列表中
        for i in range(layers):
            if i == 0:
                ts_blocks.append(TimestepBlock(in_channels, out_channels, ts_dims))
            else:
                ts_blocks.append(TimestepBlock(out_channels, out_channels, ts_dims))
        self.ts_blocks = nn.ModuleList(ts_blocks)

        if attention:
            attention_blocks = []
            for i in range(layers):
                attention_blocks.append(Attention(out_channels))
            self.attention_blocks = nn.ModuleList(attention_blocks)

        self.final_conv = Conv3x3(out_channels, out_channels, act=False)
        self.final_act = torch.nn.SiLU()

        # 如果输入输出通道数不相等，则追加一个 skip conv
        if in_channels != out_channels:
            self.skip_conv = Conv1x1(in_channels, out_channels)

        # downscale 和 upsacle
        if downscale:
            self.ds = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        if upsacle:
            self.us = lambda x: nn.functional.interpolate(x, scale_factor=2, mode="nearest")

    def forward(self, x, t):
        input_x = x
        input_t = t

        if self.attention:
            for res, atten in zip(self.ts_blocks, self.attention_blocks):
                input_x = res(input_x, input_t)
                input_x = atten(input_x)
        else:
            for res in self.ts_blocks:
                input_x = res(input_x, input_t)

        skip_x = x
        if self.skip:
            skip_x = self.skip_conv(x)

        final_x = input_x + skip_x
        final_x = self.final_act(final_x)

        if self.downscale:
            final_x = self.ds(final_x)
        if self.upscale:
            final_x = self.us(final_x)

        return final_x
