import cv2
import torch
import numpy as np

from app_utils import *

model = torch.load("model.pth")

cap = cv2.VideoCapture(0)

while True:
  ret, frame = cap.read()

  processed_frame = label_eyes(prepare_frame(frame), model)

  cv2.imshow('Label Eyes', processed_frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
