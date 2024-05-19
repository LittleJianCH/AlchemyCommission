import os
import torch
import numpy as np

from torch import nn
from torch import optim
from torchvision import datasets
from torchvision import transforms

from model import FaceNet
from dataset import FaceDB_Dataset
from const import *


def train(train_data): 
  model = FaceNet()
  loss = nn.MSELoss()
  opt = optim.Adagrad(model.parameters(), lr=0.01)

  for epoch in range(n_epochs):
    for i, (xs, ys) in enumerate(train_data):
      opt.zero_grad()
      preds = model(xs)
      l = loss(preds, ys)
      l.backward()
      opt.step()

      if i % 10 == 0:
        print(f"Epoch {epoch}, iter {i}, loss: {l.item()}")
        print(preds.detach().numpy(), ys.detach().numpy(), model(xs).detach().numpy())
    
  return model

if __name__ == "__main__":
  with open("train.txt", "r") as f:
    ids = eval(f.readline())
  
  train_set = FaceDB_Dataset("../datasets/BioID-FaceDatabase", ids, transform=None)

  model = train(train_set)
  torch.save(model, "model.pth")