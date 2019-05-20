import cv2
import numpy as np
import os
import tables
def getnum(n, max):
	if (n==0 and max == 1) :
		return str(0)
	if(n < max):
		return str(0) + getnum(n, max//10)
	else:
		return str(n)

sample = np.zeros((256,256))
hdf5_path = "mask.hdf5"
hdf5_file = tables.open_file(hdf5_path, mode = 'w')

filters = tables.Filters(complevel = 5, complib = 'blosc')
earray = hdf5_file.create_earray(
	hdf5_file.root,
	'data', # 数据名称，之后需要通过它来访问数据
	tables.Atom.from_dtype(sample.dtype), # 设定数据格式（和data1格式相同）
	shape=(0, 256), # 第一维的 0 表示数据可沿行扩展
	filters=filters,
	#expectedrows=20000 # 完整数据大约规模，可以帮助程序提高时空利用效率
)
hdf5_file.close()

hdf5_file = tables.open_file(hdf5_path, mode = 'a')
hdf5_data = hdf5_file.root.data

for n in range(12000):
	print(n)
	name = getnum(n, 10000) 
	#print(name)
	img_in = cv2.imread(name+'.png')
	size = img_in.shape
	#print(size)

	img_out = np.zeros((256, 256))

	for i in range(256):
		for j in range(256):
			if((img_in[i*2][j*2][0] == 0) and (img_in[i*2 + 1][j*2][0] == 0) and (img_in[i*2][j*2+1][0] == 0) and (img_in[i*2+ 1][j*2+1][0] == 0)):
				img_out[i][j] = 1
			else:
				img_out[i][j] = 0

	hdf5_data.append(img_out)

hdf5_file.close()
