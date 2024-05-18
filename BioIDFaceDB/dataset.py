import torch

from torchvision import datasets
from PIL import Image
import numpy as np

class FaceDB_Dataset(torch.utils.data.Dataset):
  def __init__(self, root, transform=None):
    self.root = root
    self.transform = transform

  def __getitem__(self, idx):
    name = f"BioID_{idx:04d}"

    img = Image.open(f"{self.root}/{name}.pgm")
    if self.transform:
      img = self.transform(img)

    with open(f"{self.root}/{name}.eye") as f:
      f.readline() # skip the first line
      line = f.readline()
      labels = list(map(int, line.split()))
    
    return torch.from_numpy(np.array(img)), np.array(labels)

  def __len__(self):
    return 1520 # hardcoded for now