import os
import random

from const import *

dataset_dir = "../datasets/BioID-FaceDatabase"

ids = []
for file in os.listdir(dataset_dir):
  if file.endswith(".pgm"):
    ids.append(int(file.split("_")[1].split(".")[0]))

n = len(ids)
train_size = int(trainset_portain * n)

random.shuffle(ids)
train_set = ids[:train_size]
test_set = ids[train_size:]

def write_into_file(file, ids):
  with open(file, "w") as f:
    f.write(f"{ids}")

write_into_file("train.txt", train_set)
write_into_file("test.txt", test_set)
