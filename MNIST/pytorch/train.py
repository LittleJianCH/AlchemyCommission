import torch

from torch import nn
from torch import optim
from torchvision import datasets
from torchvision import transforms

from const import *
from model import MLP

train_dataset = datasets.MNIST(
  root=data_path, train=True, download=True, transform=transforms.ToTensor()
)
train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def train():
  model = MLP()
  loss = nn.CrossEntropyLoss()
  opt = optim.SGD(model.parameters(), lr=0.01)

  for epoch in range(n_epochs):
    for i, (xs, ys) in enumerate(train_data):
      opt.zero_grad()
      preds = model(xs)
      l = loss(preds, ys)
      l.backward()
      opt.step()

      if i % 100 == 0:
        print(f"Epoch {epoch}, iter {i}, loss: {l.item()}")
    
  return model

if __name__ == "__main__":
  model = train()
  torch.save(model, "model.pth")