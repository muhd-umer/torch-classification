import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import unittest

import torch

from config import get_config
from models import EfficientNetV2, MBConv, MBConvConfig, get_structure

cfg = get_config()
residual_config = [
    MBConvConfig(*layer_config) for layer_config in get_structure(cfg.model_name)
]


class TestEfficientNetV2(unittest.TestCase):
    def setUp(self):
        self.layer_infos = residual_config
        self.model = EfficientNetV2(self.layer_infos, num_classes=10)

    def test_forward(self):
        x = torch.randn(1, 3, 224, 224)
        out = self.model(x)
        self.assertEqual(out.shape, (1, 10))

    def test_change_dropout_rate(self):
        self.model.change_dropout_rate(0.5)
        self.assertEqual(self.model.head[-2].p, 0.5)

    def test_make_layers(self):
        layers = self.model.make_layers(self.layer_infos[0], MBConv)
        self.assertEqual(len(layers), self.layer_infos[0].num_layers)

    def test_get_sd_prob(self):
        sd_prob = self.model.get_sd_prob()
        self.assertEqual(
            sd_prob,
            self.model.stochastic_depth * (self.model.cur_block / self.model.num_block),
        )


if __name__ == "__main__":
    unittest.main()
