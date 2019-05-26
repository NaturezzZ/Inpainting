import sys
import cv2
root = sys.argv[1]
crop_size = (256, 256)
img = cv2.imread(root)
#print(img)
img_new = cv2.resize(img, crop_size, interpolation = cv2.INTER_CUBIC)
cv2.imwrite('newpic.png',img_new)