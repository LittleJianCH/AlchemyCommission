import cv2
import torch
import numpy as np

def prepare_frame(frame):
  target_w = 288
  target_h = 384

  resize_ratio = max(target_w / frame.shape[0], target_h / frame.shape[1])
  resized_frame = cv2.resize(frame, (0, 0), fx=resize_ratio, fy=resize_ratio)

  x = (resized_frame.shape[0] - target_w) // 2
  y = (resized_frame.shape[1] - target_h) // 2

  cropped_frame = resized_frame[x:x + target_w, y:y + target_h]

  gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

  return gray_frame

def mark_eye(image, eye, color):
  cv2.circle(image, (eye[0], eye[1]), 7, color, 1)
  cv2.circle(image, (eye[2], eye[3]), 7, color, 1)

def label_eyes(frame, model):
  xs = torch.from_numpy(np.array(frame)).type(torch.FloatTensor).unsqueeze(0)
  pred_eyes = list(map(int, model(xs).detach().numpy()))
  
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
  mark_eye(rgb_frame, pred_eyes, (0, 0, 255))

  return rgb_frame