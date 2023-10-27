from models.unconditional_ddpm.unet import UNet
from config.config import Config
from visualize.plot import *
from typing import Optional
import torch
from scheduler.ddpm_scheduler import DDPMScheduler


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


if __name__ == "__main__":
    config = Config(r"config.yaml")
    model = UNet(config).to(config.device)
    model.eval()

    scheduler = DDPMScheduler(config)

    # 读取模型
    checkpoint = torch.load(r"checkpoints\model_ep125")
    model.load_state_dict(checkpoint['model_state_dict'])

    image = inference(model, scheduler, config.num_inference_images, config)
    image = (image / 2 + 0.5).clamp(0, 1)
    plot_images(image, save_dir="test")
