
from typing import TYPE_CHECKING
import numpy
import torch

if TYPE_CHECKING:
    from config.config import Config


class Scheduler:
    def __init__(self, config: 'Config') -> None:
        self.config = config

        # 设置训练和推理的步数
        self.num_train_timesteps: int = self.config.num_train_timesteps
        self.num_inference_steps: int = self.config.num_inference_timesteps
        self.set_timesteps(self.num_inference_steps)        # 初始化 timesteps

        # 创建从 start 到 end 的一个 beta 的数组
        self.beta_start: float = self.config.beta_start
        self.beta_end: float = self.config.beta_end

        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps, dtype=torch.float32).to(self.config.device)

        self.alphas = 1.0 - self.betas      # alpha = 1 - beta
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)     # alphas_cumprod 即 α_bar

    def set_timesteps(self, inference_steps: int):
        '''
        生成一个时间序列的数组，

        例如：当 inference step 设为 100, train step 为 1000 时，
        则生成一个 [990, 980, 970, ..., 0] 的数组
        '''
        self.num_inference_steps = inference_steps

        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (numpy.arange(0, self.num_inference_steps) * step_ratio).round()[::-1].copy().astype(numpy.int64)

        self.inf_timesteps = torch.from_numpy(timesteps).to(self.config.device)

    def add_noise(self,
                  image: torch.Tensor,
                  noise: torch.Tensor,
                  timesteps: torch.Tensor) -> torch.Tensor:
        '''
        根据采样的时间点，对图像进行加噪

        x_t = √(α_t)x_0 + √(1-α_t) ε
        '''
        sqrt_alpha_prod = torch.sqrt(self.alphas_cumprod[timesteps])     # √α_bar_t
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(image.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = torch.sqrt((1 - self.alphas_cumprod[timesteps]))     # √1-α_bar_t
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(image.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * image + sqrt_one_minus_alpha_prod * noise

        return noisy_samples

    def sample_timesteps(self, size: int):
        ''' 
        采样一些随机的时间点
        '''
        timesteps = torch.randint(0, self.num_train_timesteps, (size,), device=self.config.device).long()
        return timesteps

    def prev_timestep(self, timestep: torch.Tensor):
        ''' 
        获取 timestep t 的前一个时间点，即 t-Δt
        Δt = num_train / num_inf
        '''
        return timestep - self.num_train_timesteps // self.num_inference_steps

    def step(self, noise_pred: torch.Tensor, timestep: torch.Tensor, noisy_image: torch.Tensor):
        ...
