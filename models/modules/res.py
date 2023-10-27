import torch.nn as nn
import torch
from models.modules.common import Conv3x3, Conv1x1
from models.modules.embeddings import TimestepBlock
from models.modules.attention import Attention
from typing import Optional


class ResnetBlock(nn.Module):
    '''
    一个不带 timestep embedding 的 resnet block
    '''

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ResnetBlock, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.skip = in_channels == out_channels

        if self.skip:
            self.conv_skip = Conv1x1(in_channels, out_channels)

    def forward(self, x):
        input_x = x
        conv1 = self.conv(x)
        if self.skip:
            input_x = self.conv_skip(input_x)
        return conv1 + input_x


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
