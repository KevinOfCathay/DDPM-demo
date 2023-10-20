import torch
from typing import TYPE_CHECKING
from thop import profile

if TYPE_CHECKING:
    from config.config import Config


class Model(torch.nn.Module):
    def __init__(self, config: 'Config') -> None:
        super(Model, self).__init__()
        self.config = config

    def model_thops(self):
        with torch.no_grad():
            x = torch.randn(1, self.config.input_channels, self.config.train_image_size, self.config.train_image_size)
            t = torch.tensor([1000])
            macs, params = profile(self, inputs=(x, t))  # type: ignore
            print("模型信息:", "MACs", macs, "Params", params)

    def to_onnx(self, path: str):
        '''
        将模型转换为 onnx
        '''
        with torch.no_grad():
            x = torch.randn(1, self.config.input_channels, self.config.train_image_size, self.config.train_image_size).to(self.config.device)
            t = torch.tensor([1000]).to(self.config.device)
            torch.onnx.export(self, (x, t), path)
