if __name__ == "__main__":
    from pathlib import Path
    import sys
    import os
    top_dir = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(top_dir))

    from models.unconditional_ddpm.unet import UNet
    from config.config import Config
    from visualize.plot import *
    from models.unconditional_ddpm.inference import inference
    import torch
    from scheduler.ddpm_scheduler import DDPMScheduler


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