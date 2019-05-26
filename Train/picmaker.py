import numpy as np
import os
import random
import tables
path = "E:\\0NaturezzZ\\SchoolWork\\2019Spring\\AI_Intro\\CV_Smallclass\\Project\\Inpainting"

def load_pic():
	print("Successfully loaded!")
	global present_path 
	present_path = os.getcwd()
	os.chdir(path)
	global mask_file
	global mask_data
	global pic_data
	global pic_file
	mask_path = "mask.hdf5"
	pic_path = "pic0.hdf5"
	
	mask_file = tables.open_file(mask_path, mode='r')
	mask_data = mask_file.root.data

	pic_file = tables.open_file(pic_path, mode='r')
	pic_data = pic_file.root.data

def check_pic():
	print("check stage")
	pic_num = random.randint(0, 35000)
	mask_num = random.randint(0, 1500)
	pic = pic_data[pic_num * 256: (pic_num + 1) * 256]
	mask = mask_data[mask_num * 256: (mask_num + 1) * 256]
	output = np.zeros((1, 256, 256, 9))
	output[:,:,:,:3] = pic.reshape(-1,256,256,3)
	output[:,:,:,3:4] = mask.reshape((-1,256,256,1))
	output[:,:,:,4:5] = output[:,:,:,3:4]
	output[:,:,:,5:6] = output[:,:,:,3:4]
	output[:,:,:,6:9] = output[:,:,:,:3] * output[:,:,:,3:6]
	return output

def makepic():
	print('Preparing for ground truth, mask and ground truth with mask...')	
	output = np.zeros((1000, 256, 256, 9))
	r = random.randint(0, 28)
	randommask = random.randint(0,5)
	print("Using picture %d and mask %d" %(r, randommask))
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
	np.random.shuffle(mask_data1)

	output[:, :, :, 3] = mask_data1[:1000, :, :, 0]
	output[:, :, :, 4] = mask_data1[:1000, :, :, 0]
	output[:, :, :, 5] = mask_data1[:1000, :, :, 0]
	output[:, :, :, 6:9] = output[:, :, :, :3] * output[:, :, :, 3:6] 

	print('Input file prepared!')
	return output
def cl_file():
	mask_file.close()
	pic_file.close()
	#print(output)
	os.chdir(present_path)
	print("file closed")