import cv2
import tkinter as tk

from PIL import Image, ImageTk

from app_utils import *

max_data_idx = 1520
datadir = "../datasets/BioID-FaceDatabase"

paused = False

cur_img = None

eyes = []

def log_data():
  global max_data_idx, cur_img

  max_data_idx = max_data_idx + 1
  filename = f"{datadir}/BioID_{max_data_idx:04d}"

  print(f"Logging data to {filename}")

  with open(f"{filename}.eye", "w") as f:
    f.writelines([
      "#LX	LY	RX	RY\n", 
      f"{eyes[0][0]}	{eyes[0][1]}	{eyes[1][0]}	{eyes[1][1]}\n"
    ])
  
  cur_img.save(f"{filename}.pgm")

def on_mouse_click(event):
  global paused, eyes
  if not paused:
    paused = True
  else:
    eyes.append((event.x, event.y))
    if len(eyes) == 2:
      log_data()
      eyes = []
      paused = False

def show_camera():
  cap = cv2.VideoCapture(0)

  root = tk.Tk()
  root.title("Camera Feed")

  root.bind("<Button-1>", on_mouse_click)

  label = tk.Label(root)
  label.pack()

  def update_frame():
    global cur_img
    ret, frame = cap.read()

    if ret and not paused:
      cur_img = prepare_frame(frame)
      cur_img = Image.fromarray(cur_img)
      photo = ImageTk.PhotoImage(image=cur_img)
      label.img = photo
      label.config(image=photo)

    label.after(5, update_frame)

  update_frame()

  root.mainloop()

  cap.release()
  cv2.destroyAllWindows()

show_camera()
