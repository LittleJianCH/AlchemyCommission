import torch

from torchvision import datasets
from PIL import Image
import numpy as np

class FaceDB_Dataset(torch.utils.data.Dataset):
  def __init__(self, root, ids, transform=None):
    self.root = root
    self.ids = ids
    self.transform = transform

  def __getitem__(self, idx):
    idx = self.ids[idx]
    name = f"BioID_{idx:04d}"

    img = Image.open(f"{self.root}/{name}.pgm")
    if self.transform:
      img = self.transform(img)

    with open(f"{self.root}/{name}.eye") as f:
      f.readline() # skip the first line
      line = f.readline()
      labels = list(map(int, line.split()))

    img_tensor = torch.from_numpy(np.array(img)).type(torch.FloatTensor).unsqueeze(0)
    label_tensor = torch.from_numpy(np.array(labels)).type(torch.FloatTensor)

    # print(img_tensor, label_tensor)
    return img_tensor, label_tensor

  def __len__(self):
    return len(self.ids)