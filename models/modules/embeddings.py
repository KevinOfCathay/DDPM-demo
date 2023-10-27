import torch
import torch.nn as nn


def timestep(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2

    maxp = torch.tensor(max_period, device=timesteps.device)
    ex = -torch.log(maxp) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
    ex = ex / half  # diffuser 中使用的是 half - downscale_freq_shift [1]

    emb = torch.exp(ex)
    emb = timesteps[:, None].float() * emb[None, :]     # [bs, emb]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # pad
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


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


# 仅测试用
if __name__ == "__main__":
    import sys
    sys.path.append(r"")

    from visualize.plot import plot_arrays

    plot_arrays(torch.arange(0, 128),
                torch.concat([timestep(torch.tensor([100]), 128),
                              timestep(torch.tensor([80]), 128),
                              timestep(torch.tensor([60]), 128),
                              timestep(torch.tensor([40]), 128),
                              timestep(torch.tensor([20]), 128),
                              ], dim=0)
                )
