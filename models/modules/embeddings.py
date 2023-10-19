import torch
import torch.nn as nn


def timestep(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2

    mp = torch.tensor(max_period, device=timesteps.device)
    ex = -torch.log(mp) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
    ex = ex / half  # diffuser 中使用的是 half - downscale_freq_shift [1]

    emb = torch.exp(ex)
    emb = timesteps[:, None].float() * emb[None, :]     # [bs, emb]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # pad
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


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
