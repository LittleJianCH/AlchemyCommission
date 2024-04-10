import torch

from torch import nn

class MLP(nn.Module):
  def __init__(self):
    super().__init__()

    self.layer1 = nn.Linear(28 * 28, 512)
    self.layer2 = nn.Linear(512, 128)
    self.layer3 = nn.Linear(128, 10)


  def forward(self, xs):
    xs = xs.view(-1, 28 * 28)
    xs = torch.relu(self.layer1(xs))
    xs = torch.relu(self.layer2(xs))
    xs = self.layer3(xs)
    return torch.softmax(xs, dim=1)
