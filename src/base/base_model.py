"""
this is nn.Module class with some basic functions
"""

import torch.nn as nn
from abc import abstractmethod
import logging


class BaseModel(nn.Module):
    def __init__(self, name: str):
        super(BaseModel, self).__init__()
        self.name = name
        self.logger = logging.getLogger(f"models.{name}")
        logging.info(f"Initializing {self.name} model")

    def total_params(self):
        return sum(p.numel() for p in self.parameters())

    def total_params_formatted(self):
        total_params = self.total_params()
        return (
            f"{total_params / 10**6:.1f}m"
            if total_params > 10**6
            else f"{total_params / 10**3:.1f}k"
        )

    @abstractmethod
    def load_pretrained(self, pretrained_model: "BaseModel"):
        """
        Load pretrained model parameters into current model
        """
        pass
