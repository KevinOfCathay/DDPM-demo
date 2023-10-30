from typing import Optional
import torch
from config.config import Config


def inference(model, scheduler, images: int, config: 'Config', noise: Optional[torch.Tensor] = None):
    # 选择使用固定的噪声来推理，还是随机的噪声来推理
    if noise is None:
        noisy_sample = torch.randn((images, config.input_channels, config.train_image_size,  config.train_image_size)).to(config.device)
    else:
        noisy_sample = noise

    for t in scheduler.inf_timesteps:
        with torch.no_grad():   # 不加入这一行显存会溢出
            noisy_pred = model(noisy_sample, t[None].to(config.device)).sample
            noisy_sample = scheduler.step(noisy_pred, t, noisy_sample)  # type: ignore
    return noisy_sample