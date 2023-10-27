import torch
import torch.nn as nn
from models.modules.common import Conv3x3
from models.modules.res import ResnetBlock
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config.config import Config


class Encoder(nn.Module):
    def __init__(self,
                 config: 'Config',
                 channels=(64, 64, 128, 128),
                 attention: bool = False) -> None:
        super(Encoder, self).__init__()

        self.channels = channels
        self.res_list = nn.ModuleList()

        # 输入通道数
        in_channels = config.input_channels
        for channel in channels:
            res = ResnetBlock(in_channels, channel)
            self.res_list.append(res)

            in_channels = channel   # 输出的通道数作为下一层的输入通道数

    def forward(self, x):
        pass
