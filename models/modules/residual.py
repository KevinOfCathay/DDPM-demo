import torch.nn as nn
import torch
from models.modules.common import Conv3x3, Conv1x1, AvgPool2x, Upsample2x
from models.modules.attention import Attention


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embed_dims: int):
        '''
        channels: 输入/输出通道数
        '''
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dims = embed_dims

        self.conv1 = Conv3x3(in_channels, out_channels)
        self.conv2 = Conv3x3(out_channels, out_channels)
        self.conv_skip = Conv1x1(in_channels, out_channels)

    def forward(self, x: torch.Tensor, ts: torch.Tensor):
        '''
        x: 输入图片
        t: timestep
        '''
        x_conv_1 = self.conv1(x)

        x_add = torch.add(x_conv_1, ts)
        x_conv_2 = self.conv2(x_add)

        x_conv_skip = self.conv_skip(x)
        x_final = torch.add(x_conv_skip, x_conv_2)
        return x_final


class ResAttentionModule(nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int, ts_embed_dims: int, layers: int = 1,
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

        self.ts_projection = nn.Linear(ts_embed_dims, out_channels)

        res_blocks = []
        if attention:
            attention_blocks = []

        # 将 block 加入到列表中
        for i in range(layers):
            if i == 0:
                res_blocks.append(ResBlock(in_channels, out_channels, ts_embed_dims))
            else:
                res_blocks.append(ResBlock(out_channels, out_channels, ts_embed_dims))
            if attention:
                attention_blocks.append(Attention(out_channels))

        # 创建 ModuleList
        self.res_blocks = nn.ModuleList(res_blocks)
        if attention:
            self.attention_blocks = nn.ModuleList(attention_blocks)

        self.final_block = ResBlock(out_channels, out_channels, ts_embed_dims)

        # downscale 和 upsacle
        if downscale:
            self.ds = AvgPool2x()
        if upsacle:
            self.us = Upsample2x(out_channels, out_channels, True)

    def forward(self, x, t):
        input_x = x
        input_t = self.ts_projection(t)[:, :, None, None]

        if self.attention:
            for res, atten in zip(self.res_blocks, self.attention_blocks):
                input_x = res(input_x, input_t)
                input_x = atten(input_x)
        else:
            for res in self.res_blocks:
                input_x = res(input_x, input_t)

        # 最后再加一个 res block
        input_x = self.final_block(input_x, input_t)

        if self.downscale:
            input_x = self.ds(input_x)
        if self.upscale:
            input_x = self.us(input_x)

        return input_x
