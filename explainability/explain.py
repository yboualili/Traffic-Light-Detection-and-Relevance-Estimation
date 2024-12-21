from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image
import cv2
from ultralytics import YOLO
import numpy as np


model = YOLO('../models/epoch35.pt')
target_layers = [model.model.model[-3]]
cam = EigenCAM(model, target_layers,task='od')

img = cv2.imread('test.jpg')
img = cv2.resize(img, (1920,1216))
rgb_img = img.copy()
img = np.float32(img) / 255

grayscale_cam = cam(img)[0, :, :]
cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

cv2.imwrite('test_cam.jpg', cam_image)