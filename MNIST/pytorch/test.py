import torch

from torchvision import datasets
from torchvision import transforms

from const import *
from model import MLP

test_dataset = datasets.MNIST(
  root=data_path, train=False, download=True, transform=transforms.ToTensor()
)
test_data = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

def test(model):
  correct = 0
  total = 0

  with torch.no_grad():
    for i, (xs, ys) in enumerate(test_data):
      preds = model(xs)
      _, predicted = torch.max(preds, 1)
      total += ys.size(0)
      correct += (predicted == ys).sum().item()

      if i % 50 == 0 and i != 0:
        print(f"iter {i}, accuracy: {correct / total}")

  print(f"Accuracy: {correct / total}")


if __name__ == "__main__":
  model = torch.load("model.pth")
  test(model)