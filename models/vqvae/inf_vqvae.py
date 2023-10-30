if __name__ == "__main__":
    from pathlib import Path
    import sys
    top_dir = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(top_dir))

    from diffusers.models.vq_model import VQModel
    from data.data import ImageData
    from config.config import Config
    import torch
    from visualize.plot import *

    config = Config(r"config.yaml")
    
    # 数据路径
    dataset = ImageData(config, r"")
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=81, shuffle=True)

    # 读取 vae 模型
    vqvae = VQModel(
        config.input_channels, config.input_channels,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(128, 64, 32),
        latent_channels=32,
        layers_per_block=1
    ).to(config.device)
    checkpoint = torch.load(top_dir/Path(r"test/model_ep10"))
    vqvae.load_state_dict(checkpoint['model_state_dict'])

    vqvae.eval()
    for batch in train_dataloader:
        encoded = vqvae.encode(batch)
        z = encoded.latents
        generated_images = (z / 2 + 0.5).clamp(0, 1)
        plot_images(generated_images, save_dir=config.proj_name, save_title="z", cols=9)

        quantized_z, _, _ = vqvae.quantize(z)
        generated_images = (quantized_z / 2 + 0.5).clamp(0, 1)
        plot_images(generated_images, save_dir=config.proj_name, save_title="quantized_z", cols=9)

        decoded = vqvae.decode(quantized_z, force_not_quantize=True)[0]
        generated_images = (decoded / 2 + 0.5).clamp(0, 1)
        plot_images(generated_images, save_dir=config.proj_name, save_title="decoded", cols=9)
        break