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

sample = np.zeros((256,256,3))
hdf5_path = "pic.hdf5"
hdf5_file = tables.open_file(hdf5_path, mode = 'w')

filters = tables.Filters(complevel = 5, complib = 'blosc')
earray = hdf5_file.create_earray(
	hdf5_file.root,
	'data', # 数据名称，之后需要通过它来访问数据
	tables.Atom.from_dtype(sample.dtype), # 设定数据格式（和data1格式相同）
	shape=(0, 256, 3), # 第一维的 0 表示数据可沿行扩展
	filters=filters,
	#expectedrows=20000 # 完整数据大约规模，可以帮助程序提高时空利用效率
)
hdf5_file.close()

hdf5_file = tables.open_file(hdf5_path, mode = 'a')
hdf5_data = hdf5_file.root.data

for n in range(1,36500):
	print(n)
	name = 'Places365_val_000'+getnum(n, 10000) 
	#print(name)
	print(name+'.jpg')
	img_in = cv2.imread(name+'.jpg')
	size = img_in.shape
	print(size)
	hdf5_data.append(img_in)

hdf5_file.close()
