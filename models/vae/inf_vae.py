if __name__ == "__main__":
    from pathlib import Path
    import sys
    top_dir = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(top_dir))

    from diffusers.models.autoencoder_kl import AutoencoderKL
    from visualize.plot import *
    from config.config import Config
    import torch

    config = Config(r"config.yaml")

    # 读取 vae 模型
    vae = AutoencoderKL(
        config.input_channels, config.input_channels,
        ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(128, 64, 32),
        latent_channels=16,
        layers_per_block=2
    ).to(config.device)
    checkpoint = torch.load(top_dir/Path(r"checkpoints/model_ep20"))
    vae.load_state_dict(checkpoint['model_state_dict'])

    vae.eval()
    with torch.no_grad():
        noise = torch.randn(81, 16, 16, 16).to(config.device)
        generated_images = vae.decode(noise).sample
        generated_images = (generated_images / 2 + 0.5).clamp(0, 1)
        plot_images(generated_images, save_dir=config.proj_name, save_title="vae_decode", cols=9)
