import torch
import torch.nn as nn
from models.modules.common import Conv3x3, Concat
from models.modules.residual import ResAttentionModule
from models.modules.embeddings import timestep
from typing import TYPE_CHECKING
from models.model import Model

if TYPE_CHECKING:
    from config.config import Config

# 模型架构有问题，暂时无法使用
# 目前尚还不确定问题出在哪里
class UNet(Model):
    def __init__(self, config: 'Config') -> None:
        super(UNet, self).__init__(config)

        ch = config.model_channels
        self.ch_input = config.input_channels
        layers = config.layers
        ts_embed_dims = config.ts_embed_dims

        self.transform = nn.Sequential(nn.Linear(ts_embed_dims, ts_embed_dims), nn.SiLU())

        self.conv_L1 = Conv3x3(self.ch_input, ch, 1)      # 3 -> 64

        self.res_atten_L1 = ResAttentionModule(ch, ch, ts_embed_dims, attention=False, layers=layers, downscale=True)      # 64 x 32 x 32 -> 64 x 16 x 16
        self.res_atten_L2 = ResAttentionModule(ch, ch*2, ts_embed_dims, attention=False, layers=layers)                             # 64 x 16 x 16 -> 128 x 16 x 16
        self.res_atten_L3 = ResAttentionModule(ch*2, ch*2, ts_embed_dims, attention=False, layers=layers, downscale=True)  # 128 x 16 x 16 -> 128 x 8 x 8
        self.res_atten_L4 = ResAttentionModule(ch*2, ch*4, ts_embed_dims, attention=False, layers=layers)                           # 128 x 8 x 8 -> 256 x 8 x 8
        self.res_atten_L5 = ResAttentionModule(ch*4, ch*4, ts_embed_dims, attention=False, layers=layers, downscale=True)  # 256 x 8 x 8 -> 256 x 4 x 4
        self.res_atten_L6 = ResAttentionModule(ch*4, ch*4, ts_embed_dims, attention=False, layers=layers)                           # 256 x 4 x 4 -> 512 x 4 x 4

        self.res_atten_M1 = ResAttentionModule(ch*4, ch*4, ts_embed_dims, attention=False, layers=layers)                           # 512 x 4 x 4 -> 512 x 4 x 4

        self.concat = Concat()

        self.res_atten_R1 = ResAttentionModule(ch*4, ch*4, ts_embed_dims, attention=False, layers=layers)                           # 512 x 4 x 4 -> 256 x 4 x 4
        self.res_atten_R2 = ResAttentionModule(ch*8, ch*4, ts_embed_dims, attention=False, layers=layers, upsacle=True)    # 512 x 4 x 4 -> 256 x 8 x 8
        self.res_atten_R3 = ResAttentionModule(ch*4, ch*4, ts_embed_dims, attention=False, layers=layers)                           #  256 x 8 x 8 -> 256 x 8 x 8
        self.res_atten_R4 = ResAttentionModule(ch*8, ch*2, ts_embed_dims, attention=False, layers=layers, upsacle=True)   # 512 x 8 x 8 -> 128 x 16 x 16
        self.res_atten_R5 = ResAttentionModule(ch*2, ch*2, ts_embed_dims, attention=False, layers=layers)                           # 128 x 16 x 16 -> 128 x 16 x 16
        self.res_atten_R6 = ResAttentionModule(ch*4, ch, ts_embed_dims, attention=False, layers=layers, upsacle=True)       # 256 x 16 x 16 -> 64 x 32 x 32
        self.res_atten_R7 = ResAttentionModule(ch*2, ch, ts_embed_dims, attention=False, layers=layers)                               # 128 x 32 x 32 -> 64 x 32 x 32

        self.conv_l19 = Conv3x3(ch, self.ch_input, 1)       # 64 -> 3

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
        ts = self.transform(ts)

        conv_1 = self.conv_L1(x)    # 3->64
        res_l1 = self.res_atten_L1(conv_1, ts)
        res_l2 = self.res_atten_L2(res_l1, ts)
        res_l3 = self.res_atten_L3(res_l2, ts)
        res_l4 = self.res_atten_L4(res_l3, ts)
        res_l5 = self.res_atten_L5(res_l4, ts)
        res_l6 = self.res_atten_L6(res_l5, ts)

        res_m1 = self.res_atten_M1(res_l6, ts)
        res_r1 = self.res_atten_R1(res_m1, ts)

        # copy and concat
        concat_r1_l4 = self.concat(res_r1, res_l5)
        res_r2 = self.res_atten_R2(concat_r1_l4, ts)
        res_r3 = self.res_atten_R3(res_r2, ts)

        # copy and concat
        concat_8_4 = self.concat(res_r3, res_l4)
        res_r4 = self.res_atten_R4(concat_8_4, ts)
        res_r5 = self.res_atten_R5(res_r4, ts)

        # copy and concat
        concat_9_3 = self.concat(res_r5, res_l2)
        res_r6 = self.res_atten_R6(concat_9_3, ts)

        # copy and concat
        concat_10_2 = self.concat(res_r6, conv_1)
        res_r7 = self.res_atten_R7(concat_10_2, ts)

        # copy and concat
        final = self.conv_l19(res_r7)

        return final

    def to_onnx(self, path: str):
        '''
        将模型转换为 onnx
        '''
        with torch.no_grad():
            x = torch.randn(1, self.ch_input, self.config.train_image_size, self.config.train_image_size)
            t = torch.tensor([1000])
            torch.onnx.export(self, (x, t), path)
