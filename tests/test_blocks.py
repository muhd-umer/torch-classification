import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import unittest

import torch
import torch.nn as nn

from models import ConvBNAct, SEUnit, StochasticDepth


class TestSEUnit(unittest.TestCase):
    def setUp(self):
        self.in_channel = 32
        self.reduction_ratio = 4
        self.se_unit = SEUnit(self.in_channel, self.reduction_ratio)

    def test_forward(self):
        x = torch.randn(1, self.in_channel, 224, 224)
        out = self.se_unit(x)
        self.assertEqual(out.shape, x.shape)

    def test_avg_pool(self):
        x = torch.randn(1, self.in_channel, 224, 224)
        out = self.se_unit.avg_pool(x)
        self.assertEqual(out.shape, (1, self.in_channel, 1, 1))

    def test_fc(self):
        x = torch.randn(1, self.in_channel, 1, 1)
        out = self.se_unit.fc[0](x)
        self.assertEqual(out.shape, (1, self.in_channel // self.reduction_ratio, 1, 1))
        out = self.se_unit.fc[1](out)
        self.assertEqual(out.shape, (1, self.in_channel, 1, 1))


class TestConvBNAct(unittest.TestCase):
    def setUp(self):
        self.in_channel = 3
        self.out_channel = 32
        self.kernel_size = 3
        self.stride = 1
        self.groups = 1
        self.norm_layer = nn.BatchNorm2d
        self.act = nn.ReLU
        self.conv_layer = nn.Conv2d
        self.conv_bn_act = ConvBNAct(
            self.in_channel,
            self.out_channel,
            self.kernel_size,
            self.stride,
            self.groups,
            self.norm_layer,
            self.act,
            self.conv_layer,
        )

    def test_forward(self):
        x = torch.randn(1, self.in_channel, 224, 224)
        out = self.conv_bn_act(x)
        self.assertEqual(out.shape, (1, self.out_channel, 224, 224))

    def test_layers(self):
        self.assertIsInstance(self.conv_bn_act[0], nn.Conv2d)
        self.assertIsInstance(self.conv_bn_act[1], nn.BatchNorm2d)
        self.assertIsInstance(self.conv_bn_act[2], nn.ReLU)


class TestStochasticDepth(unittest.TestCase):
    def setUp(self):
        self.prob = 0.5
        self.mode = "row"
        self.stochastic_depth = StochasticDepth(self.prob, self.mode)

    def test_forward(self):
        x = torch.randn(1, 3, 224, 224)
        out = self.stochastic_depth(x)
        self.assertEqual(out.shape, x.shape)

    def test_forward_no_drop(self):
        self.stochastic_depth.prob = 0.0
        x = torch.randn(1, 3, 224, 224)
        out = self.stochastic_depth(x)
        self.assertTrue(torch.equal(out, x))

    def test_forward_not_training(self):
        self.stochastic_depth.training = False
        x = torch.randn(1, 3, 224, 224)
        out = self.stochastic_depth(x)
        self.assertTrue(torch.equal(out, x))


if __name__ == "__main__":
    unittest.main()
