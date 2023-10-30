if __name__ == "__main__":
    from pathlib import Path
    import sys
    import os
    top_dir = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(top_dir))

    from torch.utils.data import DataLoader
    from models.unconditional_ddpm.diffuser_unet import DFUNet
    from config.config import Config
    from data.mnist_data import MNISTData
    from scheduler.ddpm_scheduler import DDPMScheduler
    from visualize.plot import *
    from models.unconditional_ddpm.inf_ddpm import inference
    import torch
    import torch.nn as nn

    # 用于显示进度条，不需要可以去掉
    from tqdm.auto import tqdm

    config = Config(r"config.yaml")
    model = DFUNet(config).to(config.device)
    scheduler = DDPMScheduler(config)

    # diffusers 里面用的是 AdamW,
    # lr 不能设置的太大
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    training_data = MNISTData(config, r"dataset", return_label=True)
    train_dataloader = DataLoader(training_data, batch_size=config.batch, shuffle=True)

    # 显示原图像和加噪后的图像
    test_images = torch.concat([training_data[i][0].unsqueeze(0) for i in range(8)])
    test_labels = torch.concat([training_data[i][1].unsqueeze(0) for i in range(8)])
    timesteps = scheduler.sample_timesteps(8)
    noise = torch.randn(test_images.shape).to(config.device)
    noisy_image = scheduler.add_noise(image=test_images, noise=noise, timesteps=timesteps)
    plot_images((test_images / 2 + 0.5).clamp(0, 1), titles=test_labels.detach().tolist(), fig_titles="original image", save_dir=config.proj_name)
    plot_images((noisy_image / 2 + 0.5).clamp(0, 1), titles=test_labels.detach().tolist(), fig_titles="noisy image", save_dir=config.proj_name)

    # 训练模型
    for ep in range(config.epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        model.train()
        for image, _ in train_dataloader:
            batch = image.shape[0]
            timesteps = scheduler.sample_timesteps(batch)
            noise = torch.randn(image.shape).to(config.device)
            noisy_image = scheduler.add_noise(image=image, noise=noise, timesteps=timesteps)

            pred = model(noisy_image, timesteps)[0]
            loss = torch.nn.functional.mse_loss(pred, noise)
            optimizer.zero_grad()
            loss.backward()

            # gradient clipping, 用来防止 exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "ep": ep+1}
            progress_bar.set_postfix(**logs)

        # 保存模型
        if (ep+1) % config.save_period == 0 or (ep+1) == config.epochs:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, r"checkpoints/model_ep" + str(ep+1))

        # 采样一些图片
        if (ep+1) % config.sample_period == 0:
            model.eval()
            labels = torch.randint(0, 9, (config.num_inference_images, 1)).to(config.device)
            image = inference(model, scheduler, config.num_inference_images, config, label=labels)
            image = (image / 2 + 0.5).clamp(0, 1)
            plot_images(image, save_dir=config.proj_name, titles=labels.detach().tolist())
            model.train()
