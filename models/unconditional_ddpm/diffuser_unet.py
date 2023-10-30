from diffusers import UNet2DModel
from models.model import Model
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config.config import Config


class DFUNet(Model):
    def __init__(self, config: 'Config') -> None:
        super(DFUNet, self).__init__(config)

        # 参考
        # https://huggingface.co/docs/diffusers/tutorials/basic_training
        self.model = UNet2DModel(
            sample_size=config.train_image_size,  # 图像大小
            in_channels=config.input_channels,  # 输入通道数
            out_channels=config.input_channels,  # 输出通道数
            layers_per_block=config.layers,
            block_out_channels=(64, 64, 128, 128, 256, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D",
                              "DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D",  "UpBlock2D", "UpBlock2D",
                            "UpBlock2D", "UpBlock2D", "UpBlock2D")
        )

    def forward(self, x, ts):
        return self.model(x, ts)[0]