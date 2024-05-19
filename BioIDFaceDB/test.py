import tkinter as tk
from PIL import Image, ImageTk
import cv2
import torch
import numpy as np

dir_path = "../datasets/BioID-FaceDatabase"
model = torch.load("model.pth")

def read_eye(path):
  with open(path, "r") as f:
    f.readline()
    line = f.readline()
    return list(map(int, line.split()))

def mark_eye(image, eye, color):
  cv2.circle(image, (eye[0], eye[1]), 7, color, 1)
  cv2.circle(image, (eye[2], eye[3]), 7, color, 1)

def get_photo(id):
  file_name = f"BioID_{id:04d}"

  image = cv2.imread(f"{dir_path}/{file_name}.pgm")

  labeled_eyes = read_eye(f"{dir_path}/{file_name}.eye")
  mark_eye(image, labeled_eyes, (0, 255, 0))
  
  xs = torch.from_numpy(np.array(Image.open(f"{dir_path}/{file_name}.pgm"))).type(torch.FloatTensor).unsqueeze(0)
  pred_eyes = list(map(int, model(xs).detach().numpy()))
  mark_eye(image, pred_eyes, (255, 0, 0))
  
  image_pil = Image.fromarray(image)
  photo = ImageTk.PhotoImage(image_pil)

  return photo


class DisplayPicApp:
  def __init__(self, master, ids):
    self.ids = ids
    self.cur_idx = 0

    self.image_label = tk.Label(master)
    self.image_label.pack()

    self.id_label = tk.Label(master, text="")
    self.id_label.pack()

    self.show_pic()

    self.next_button = tk.Button(master, text="Next", command=self.next_pic)
    self.next_button.pack()

    self.prev_button = tk.Button(master, text="Prev", command=self.prev_pic)
    self.prev_button.pack()
  
  def show_pic(self):
    idx = self.ids[self.cur_idx]
    photo = get_photo(idx)
    self.image_label.configure(image=photo)
    self.image_label.image = photo

    self.id_label.configure(text=f"ID: {idx}")
  
  def next_pic(self):
    self.cur_idx = (self.cur_idx + 1) % len(self.ids)
    self.show_pic()
  
  def prev_pic(self):
    self.cur_idx = (self.cur_idx - 1) % len(self.ids)
    self.show_pic()
  

if __name__ == "__main__":
  with open("train.txt", "r") as f:
    ids = eval(f.readline())
  
  root = tk.Tk()
  app = DisplayPicApp(root, ids)
  root.mainloop()
