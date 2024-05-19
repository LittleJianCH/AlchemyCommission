import sys
import os
import cv2
import torch
import numpy as np

from app_utils import *


model = torch.load("model.pth")

picname = len(sys.argv) > 1 and sys.argv[1] or "example.pgm"

image = cv2.imread(picname)

gray_img = prepare_frame(image)

processed_img = label_eyes(gray_img, model)

cv2.imshow("Image with Eyes Marked", processed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()