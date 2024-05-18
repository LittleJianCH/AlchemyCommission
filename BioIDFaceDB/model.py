import torch
import functools

from torch import nn


class FaceNet(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(
      in_channels=1, out_channels=32, kernel_size=(5, 5)
    )
    self.relu1 = nn.ReLU()
    self.avgpool = nn.AvgPool2d(kernel_size=(2, 2))

    self.conv2 = nn.Conv2d(
      in_channels=32, out_channels=64, kernel_size=(5, 5)
    )
    self.relu2 = nn.ReLU()
    self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

    self.fc1 = nn.Linear(64 * 68 * 93, 128)
    self.relu3 = nn.ReLU()
    self.fc2 = nn.Linear(128, 4)

  def forward(self, x):
    layers = [
      self.conv1, self.relu1, self.avgpool,
      self.conv2, self.relu2, self.maxpool,
      torch.flatten,
      self.fc1, self.relu3, 
      self.fc2
    ]
    
    return functools.reduce(lambda x, layer: layer(x), layers, x)