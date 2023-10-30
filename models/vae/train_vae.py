# 利用 diffusers 自带的 AutoencoderKL 训练 VAE 模型
if __name__ == "__main__":
    from pathlib import Path
    import sys
    top_dir = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(top_dir))

    from diffusers.models.autoencoder_kl import AutoencoderKL
    from config.config import Config
    from data.mnist_data import MNISTData
    from visualize.plot import *
    import torch
    import torch.nn as nn

    # 用于显示进度条，不需要可以去掉
    from tqdm.auto import tqdm

    config = Config(r"config.yaml")

    # 输入数据路径
    dataset = MNISTData (config, r"")
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch, shuffle=True)

    vae = AutoencoderKL(
        config.input_channels, config.input_channels,
        ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(128, 64, 32),
        latent_channels=16,
        layers_per_block=2
    ).to(config.device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=config.lr)

    for ep in range(20):
        progress_bar = tqdm(total=len(train_dataloader))
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            # 编码与解码
            encoded = vae.encode(batch)
            z = encoded.latent_dist.sample()
            decoded = vae.decode(z)[0]

            # 计算 reconstruction loss 和 kl loss
            rec_loss = torch.nn.functional.mse_loss(batch, decoded)
            kl_loss = encoded.latent_dist.kl().mean()
            loss = rec_loss + kl_loss * 0.0025

            loss.backward()

            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "ep": ep}
            progress_bar.set_postfix(**logs)

        # 保存模型
        if (ep+1) % config.save_period == 0 or (ep+1) == 10:
            torch.save({
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, r"checkpoints/model_ep" + str(ep+1))