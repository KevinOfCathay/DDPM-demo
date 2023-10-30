from yaml import safe_load


class Config:
    def __init__(self, config_path: str) -> None:
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))

        self.proj_name = self.config["name"]

        self.num_train_timesteps = self.config["train"]["steps"]
        self.beta_start: float = self.config["train"]["beta"][0]
        self.beta_end: float = self.config["train"]["beta"][1]
        self.clip: float = self.config["clip"]

        self.batch: int = int(self.config["train"]["batch"])
        self.train_image_size: int = int(self.config["train"]["image_size"])
        self.input_channels: int = int(self.config["train"]["image_channels"])
        self.epochs: int = int(self.config["train"]["epochs"])
        self.save_period: int = int(self.config["train"]["save_period"])
        self.sample_period: int = int(self.config["train"]["sample_period"])
        self.lr: float = self.config["train"]["lr"]

        self.num_inference_timesteps: int = self.config["inf"]["steps"]
        self.num_inference_images: int = self.config["inf"]["num_images"]

        self.device: str = self.config["device"]

        self.model_channels: int = int(self.config["model"]["base_channels"])
        self.ts_embed_dims: int = int(self.config["model"]["timestep_embed_dims"])
        self.ts_proj_dims: int = int(self.config["model"]["timestep_proj_dims"])
        self.layers: int = int(self.config["model"]["layers"])
