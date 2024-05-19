import torch
import functools

from torch import nn


class FaceNet(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(
      in_channels=1, out_channels=32, kernel_size=(3, 3)
    )
    self.relu1 = nn.ReLU()
    self.avgpool = nn.AvgPool2d(kernel_size=(2, 2))

    self.conv2 = nn.Conv2d(
      in_channels=32, out_channels=64, kernel_size=(3, 3)
    )
    self.relu2 = nn.ReLU()
    self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

    self.conv3 = nn.Conv2d(
      in_channels=64, out_channels=128, kernel_size=(3, 3)
    )
    self.relu3 = nn.ReLU()
    self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

    self.conv4 = nn.Conv2d(
      in_channels=128, out_channels=128, kernel_size=(3, 3)
    )
    self.relu4 = nn.ReLU()
    self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

    self.fc1 = nn.Linear(128 * 16 * 22, 128)
    self.relu5 = nn.ReLU()
    self.fc2 = nn.Linear(128, 20)
    self.relu6 = nn.ReLU()
    self.fc3 = nn.Linear(20, 4)

  def forward(self, x):
    layers = [
      self.conv1, self.relu1, self.avgpool,
      self.conv2, self.relu2, self.maxpool,
      self.conv3, self.relu3, self.maxpool2,
      self.conv4, self.relu4, self.maxpool3,
      torch.flatten,
      self.fc1, self.relu5,
      self.fc2, self.relu6,
      self.fc3
    ]

    return functools.reduce(lambda x, layer: layer(x), layers, x)