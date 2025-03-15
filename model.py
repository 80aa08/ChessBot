import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessNet(nn.Module):
    """Sieć neuronowa do oceny pozycji i wyboru ruchów."""
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(13, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.res_block1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128)
        )
        self.policy_head = nn.Linear(128 * 8 * 8, 4672)
        self.value_head = nn.Linear(128 * 8 * 8, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        policy_logits = self.policy_head(x)
        value = torch.clamp(self.value_head(x), -1, 1)
        return policy_logits, value