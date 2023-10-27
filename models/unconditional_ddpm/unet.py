import torch
import torch.nn as nn
from models.modules.common import Conv3x3
from models.modules.res import ResAttentionModule
from models.modules.embeddings import timestep
from typing import TYPE_CHECKING
from models.model import Model

if TYPE_CHECKING:
    from config.config import Config


class UNet(Model):
    '''
    模型模拟能力比较弱
    '''

    def __init__(self, config: 'Config') -> None:
        super(UNet, self).__init__(config)

        ch = config.model_channels
        self.ch_input = config.input_channels
        layers = config.layers
        ts_embed_dims = config.ts_embed_dims
        ts_proj_dims = config.ts_proj_dims

        self.ts_transform = nn.Sequential(nn.Linear(ts_embed_dims, ts_proj_dims), nn.SiLU(), nn.Linear(ts_proj_dims, ts_proj_dims), nn.SiLU())

        self.first_conv = Conv3x3(self.ch_input, ch, 1)      # 3 -> 64

        self.res_atten_L1 = ResAttentionModule(ch, ch, ts_proj_dims, attention=False, layers=layers, downscale=True)      # 64 x 32 x 32 -> 64 x 16 x 16
        self.res_atten_L2 = ResAttentionModule(ch, ch*2, ts_proj_dims, attention=False, layers=layers)                               # 64 x 16 x 16 -> 128 x 16 x 16
        self.res_atten_L3 = ResAttentionModule(ch*2, ch*2, ts_proj_dims, attention=False, layers=layers, downscale=True)  # 128 x 16 x 16 -> 128 x 8 x 8
        self.res_atten_L4 = ResAttentionModule(ch*2, ch*4, ts_proj_dims, attention=False, layers=layers)                           # 128 x 8 x 8 -> 256 x 8 x 8
        self.res_atten_L5 = ResAttentionModule(ch*4, ch*4, ts_proj_dims, attention=False, layers=layers, downscale=True)  # 256 x 8 x 8 -> 256 x 4 x 4
        self.res_atten_L6 = ResAttentionModule(ch*4, ch*4, ts_proj_dims, attention=False, layers=layers)                           # 256 x 4 x 4 -> [256 x 4 x 4]

        self.res_atten_M1 = ResAttentionModule(ch*4, ch*4, ts_proj_dims, layers=layers)                                                 # 256 x 4 x 4 -> 256 x 4 x 4

        self.res_atten_R1 = ResAttentionModule(ch*4, ch*4, ts_proj_dims, attention=False, layers=layers)                           # 256 x 4 x 4 -> [256 x 4 x 4]
        self.res_atten_R2 = ResAttentionModule(ch*8, ch*4, ts_proj_dims, attention=False, layers=layers, upsacle=True)    # 512 x 4 x 4 -> 256 x 8 x 8
        self.res_atten_R3 = ResAttentionModule(ch*4, ch*4, ts_proj_dims, attention=False, layers=layers)                            # 256 x 8 x 8 -> 256 x 8 x 8
        self.res_atten_R4 = ResAttentionModule(ch*8, ch*2, ts_proj_dims, attention=False, layers=layers, upsacle=True)   # 512 x 8 x 8 -> 128 x 16 x 16
        self.res_atten_R5 = ResAttentionModule(ch*2, ch*2, ts_proj_dims, attention=False, layers=layers)                           # 128 x 16 x 16 -> 128 x 16 x 16
        self.res_atten_R6 = ResAttentionModule(ch*4, ch, ts_proj_dims, attention=False, layers=layers, upsacle=True)       # 256 x 16 x 16 -> 64 x 32 x 32

        self.last_conv = Conv3x3(ch, self.ch_input, 1)       # 64 -> 3

    def forward(self, x, t):
        '''
        ### Args:
            - x: 输入图片
            - t: timestep
        '''
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.long, device=self.config.device)

        # 首先将时间转换为 timestep embed
        ts = timestep(t, self.config.ts_embed_dims)
        ts = self.ts_transform(ts)

        conv_1 = self.first_conv(x)    # 3->64
        res_l1 = self.res_atten_L1(conv_1, ts)
        res_l2 = self.res_atten_L2(res_l1, ts)
        res_l3 = self.res_atten_L3(res_l2, ts)
        res_l4 = self.res_atten_L4(res_l3, ts)
        res_l5 = self.res_atten_L5(res_l4, ts)
        res_l6 = self.res_atten_L6(res_l5, ts)

        res_m1 = self.res_atten_M1(res_l6, ts)
        res_r1 = self.res_atten_R1(res_m1, ts)

        # concat
        concat_r1_l4 = torch.concat([res_r1, res_l6], dim=1)
        res_r2 = self.res_atten_R2(concat_r1_l4, ts)
        res_r3 = self.res_atten_R3(res_r2, ts)

        # concat
        concat_8_4 = torch.concat([res_r3, res_l4], dim=1)
        res_r4 = self.res_atten_R4(concat_8_4, ts)
        res_r5 = self.res_atten_R5(res_r4, ts)

        # concat
        concat_9_3 = torch.concat([res_r5, res_l2], dim=1)
        res_r6 = self.res_atten_R6(concat_9_3, ts)

        # concat
        final = self.last_conv(res_r6)

        return final