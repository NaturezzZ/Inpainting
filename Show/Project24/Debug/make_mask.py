import cv2
import numpy as np
mask = np.zeros((256,256,1))
mask = np.loadtxt("mask.txt")
#print(mask)
mask.reshape(256,256,1)
cv2.imwrite("mask.png", mask)