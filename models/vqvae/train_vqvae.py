if __name__ == "__main__":
    from pathlib import Path
    import sys
    import os
    top_dir = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(top_dir))

    from diffusers.models.vq_model import VQModel
    from data.data import ImageData
    from config.config import Config
    import torch
    import torch.nn as nn

    # 用于显示进度条，不需要可以去掉
    from tqdm.auto import tqdm

    config = Config(r"config.yaml")
    vqvae = VQModel(
        config.input_channels, config.input_channels,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(128, 64, 32),
        latent_channels=32,
        layers_per_block=1
    ).to(config.device)
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=config.lr)

    # 输入数据路径
    dataset = ImageData(config, r"")
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch, shuffle=True)

    for ep in range(config.epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            # 编码与解码
            encoded = vqvae.encode(batch)
            z = encoded.latents
            quantized_z, loss, _ = vqvae.quantize(z)
            decoded = vqvae.decode(quantized_z, force_not_quantize=True)[0]

            # 计算 loss
            rec_loss = torch.nn.functional.mse_loss(batch, decoded)
            quant_loss = loss
            loss = rec_loss + quant_loss * 0.0025

            loss.backward()

            nn.utils.clip_grad_norm_(vqvae.parameters(), 1.0)
            optimizer.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "ep": ep}
            progress_bar.set_postfix(**logs)

        # 保存模型
        if (ep+1) % config.save_period == 0 or (ep+1) == config.epochs:
            torch.save({
                'model_state_dict': vqvae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            },os.path.join(config.proj_name, r"model_ep" + str(ep+1)))