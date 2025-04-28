import torch
import torch.nn as nn
from config import Config

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(Config.INPUT_CHANNELS, Config.CHANNELS, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(Config.CHANNELS)
        self.relu = nn.ReLU()
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(Config.CHANNELS) for _ in range(Config.RESIDUAL_BLOCKS)
        ])
        self.policy_conv = nn.Conv2d(Config.CHANNELS, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4672)
        self.value_conv = nn.Conv2d(Config.CHANNELS, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, Config.CHANNELS)
        self.value_fc2 = nn.Linear(Config.CHANNELS, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.res_blocks(x)
        p = self.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)
        v = self.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = self.relu(self.value_fc1(v))
        value = self.tanh(self.value_fc2(v))
        return policy_logits, value