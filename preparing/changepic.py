import cv2
import numpy as np
import os

def getnum(n, max):
	if (n==0 and max == 1) :
		return str(0)
	if(n < max):
		return str(0) + getnum(n, max//10)
	else:
		return str(n)

name = '00000.png'
for n in range(12000):
	name = getnum(n, 10000) 
	print(name)
	img_in = cv2.imread(name+'.png')
	size = img_in.shape
	print(size)

	img_out = np.zeros((256, 256))

	for i in range(256):
		for j in range(256):
			if((img_in[i*2][j*2][0] == 0) and (img_in[i*2 + 1][j*2][0] == 0) and (img_in[i*2][j*2+1][0] == 0) and (img_in[i*2+ 1][j*2+1][0] == 0)):
				img_out[i][j] = 0
			else:
				img_out[i][j] = 255

	np.savetxt(name+'.in',img_out)