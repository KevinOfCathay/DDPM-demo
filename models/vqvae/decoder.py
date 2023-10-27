import torch
import torch.nn as nn
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config.config import Config


class Decoder(nn.Module):
    def __init__(self, config: 'Config') -> None:
        super(Decoder, self).__init__()

    def forward(self, x):
        pass
