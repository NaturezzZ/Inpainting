import numpy as np
import os
import random
import tables
path = "E:\\0NaturezzZ\\SchoolWork\\2019Spring\\AI_Intro\\CV_Smallclass\\Project\\Inpainting"

def makepic():
	print('Preparing for ground truth, mask and ground truth with mask...')
	present_path = os.getcwd()
	os.chdir(path)
	mask_path = "mask.hdf5"
	pic_path = "pic.hdf5"
	
	mask_file = tables.open_file(mask_path, mode='r')
	mask_data = mask_file.root.data
	#print(hdf5_data.shape)

	pic_file = tables.open_file(pic_path, mode='r')
	pic_data = pic_file.root.data

	output = np.zeros((1000, 256, 256, 9))
	r = random.randint(0, 35)
	randommask = random.randint(0,9)
	'''
	起点是r*1000*256 终点是(r+1)*1000*256 -256
	'''
		
	pic_data1 = pic_data[256000*r:256000*(r+1)]
	pic_data1 = np.array(pic_data1)
	pic_data1 = pic_data1.reshape(-1,256,256,3)

	output[:, :, :, :3] = pic_data1[ :, :, :];
	mask_data1 = mask_data[256000*randommask:256000*(randommask + 1)]
	mask_data1 = np.array(mask_data1)
	mask_data1 = mask_data1.reshape(-1,256,256,1)

	output[:, :, :, 3] = mask_data1[:1000, :, :, 0]
	output[:, :, :, 4] = mask_data1[:1000, :, :, 0]
	output[:, :, :, 5] = mask_data1[:1000, :, :, 0]
	output[:, :, :, 6:9] = output[:, :, :, :3] * output[:, :, :, 3:6] 
	'''
	for i in range(1000):
		print(i)
		for j in range(256):
			for k in range(256):
				output[i][j][k][0] = pic_data[r*1000*256 + i*256 + j, k, 0]
				output[i][j][k][1] = pic_data[r*1000*256 + i*256 + j][k][1]
				output[i][j][k][2] = pic_data[r*1000*256 + i*256 + j][k][2]
		for j in range(256):
			for k in range(256):
				output[i][j][k][3] = mask_data[randommask*256 + j][k]
				output[i][j][k][4] = mask_data[randommask*256 + j][k]
				output[i][j][k][5] = mask_data[randommask*256 + j][k]
				output[i][j][k][6] = output[i][j][k][0] * output[i][j][k][3] // 255
				output[i][j][k][7] = output[i][j][k][1] * output[i][j][k][3] // 255
				output[i][j][k][8] = output[i][j][k][2] * output[i][j][k][3] // 255
		randommask += 1
	'''

	mask_file.close()
	pic_file.close()
	#print(output)
	os.chdir(present_path)
	print('Preparing completed!')
	return output
#for testing
'''
make = makepic()
gt = make[0,:,:,:3]
mask = make[0,:,:,3:6]
gt_mask = make[0,:,:,6:9]
cv2.imwrite('gt.png',gt)
cv2.imwrite('mask.png',mask)
cv2.imwrite('gt_mask.png',gt_mask)
'''