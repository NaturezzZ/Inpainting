import cv2
import numpy as np
import os
import random
import struct
mask_path = 'E:\\0NaturezzZ\\SchoolWork\\2019Spring\\AI_Intro\\CV_Smallclass\\Project\\Inpainting\\mask\\testing_mask_dataset'
img_path = 'E:\\0NaturezzZ\\SchoolWork\\2019Spring\\AI_Intro\\CV_Smallclass\\Project\\Inpainting\\val_256_bin'

def getnum(n, max):
	if (n==0 and max == 1) :
		return str(0)
	if(n < max):
		return str(0) + getnum(n, max//10)
	else:
		return str(n)

def makepic():
	output = np.zeros((256, 256, 9))
	os.chdir(img_path)
	r = random.randint(0, 36500)
	name = 'Places365_val_000' + getnum(r,10000) + '.bin'
	
	#print(output)
makepic()
