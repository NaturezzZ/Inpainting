import cv2
import os
import struct
import random
import numpy as np
import matplotlib.pyplot as plt
os.chdir(os.path.split(os.path.realpath(__file__))[0])

def get_patch(x, y, filter_size):
	tmpl = filter_size // 2
	lx = x - tmpl
	rx = x + tmpl + 1
	ly = y - tmpl
	ry = y + tmpl + 1
	if(lx < 0):
		lx = 0
	if(rx > mask.shape[0]):
		rx = mask.shape[0]
	if(ly < 0):
		ly = 0
	if(ry > mask.shape[1]):
		ry = mask.shape[1]
	a = np.array([lx,rx,ly,ry], np.int32)
	return a
	
def limited_strentch(x1, y1, x, y, filter_size):
	A2 = get_patch(x, y, filter_size)
	area = 1 - mask[A2[0]:A2[1], A2[2]:A2[3]] #当前块的非遮挡区域
	lenx = A2[1] - A2[0]
	leny = A2[3] - A2[2]
	delx = x - A2[0]
	dely = y - A2[2]
	minn = 200000000
	fx = 0
	fy = 0
	lim = min(mask.shape[0], mask.shape[1])
	Nor  = 1 / (min(mask.shape[0], mask.shape[1]) / 30)
	for t in range(8):
		dx = DX[t]
		dy = DY[t]
		nx = int(x1 + dx)
		ny = int(y1 + dy)
		cnt = 0
		while(cnt * 6 < lim and nx>=0 and nx < mask.shape[0]-lenx and ny>=0 and ny<=mask.shape[1]-leny):
			++cnt
			if (or_mask[nx:nx+lenx,ny:ny+leny].sum() == 0):
				B = get_patch(nx,ny,filter_size*2 + 1)
				a = img[:, nx:nx+lenx, ny:ny+leny] - img[:, A2[0]:A2[1], A2[2]:A2[3]]
				a = (a * area)**2
				a = (CNT[B[0]:B[1], B[2]:B[3]].sum()  + a.sum() + (abs(x1+delx - (nx + lenx // 2)) + abs(y1+dely - (ny + leny // 2)))* Nor)
				if(a < minn):
					minn = a
					fx = nx
					fy = ny
			nx += dx
			ny += dy
	A = get_patch(x1+delx, y1+dely, filter_size * 4) #x1而不是x
	for i in range(A[0], A[1] - lenx):
		for j in range(A[2], A[3] - leny):
			if(or_mask[i:i+lenx, j:j+leny].sum() == 0):
				B = get_patch(i,j,filter_size*2 + 1)
				a = img[:, i:i+lenx, j:j+leny] - img[:, A2[0]:A2[1], A2[2]:A2[3]]
				a = (a * area)**2
				a = (CNT[B[0]:B[1], B[2]:B[3]].sum() + a.sum() + (abs(x1+delx - (i + lenx // 2)) + abs(y1+dely - (j + leny // 2))) * Nor)
				if(a < minn):
					minn = a
					fx = i
					fy = j
	return fx, fy
	
def find_nearst(x, y, filter_size): #保证两块区域的形状一样,返回的不是中心坐标，而是左上角坐标！！！！
	A2 = get_patch(x, y, filter_size)
	area = 1 - mask[A2[0]:A2[1], A2[2]:A2[3]] #当前块的非遮挡区域
	lenx = A2[1] - A2[0]
	leny = A2[3] - A2[2]
	delx = x - A2[0]
	dely = y - A2[2]
	minn = 200000000
	fx = 0
	fy = 0
	Nor  = 1 / (min(mask.shape[0], mask.shape[1]) / 30)
	A = get_patch(x, y, 3)
	for px in range(A[0], A[1]):
		for py in range(A[2], A[3]):
			if(mask[px][py] == 0):
				B = get_patch(from_x[px][py], from_y[px][py], 3)
				for nx in range(B[0],B[1]):
					for ny in range(B[2],B[3]):
						nx = int(nx - delx)#中心坐标到左上角坐标
						ny = int(ny - dely)
						if (nx>=0 and nx < mask.shape[0]-lenx and ny>=0 and ny<=mask.shape[1]-leny):
							if (or_mask[nx:nx+lenx,ny:ny+leny].sum() == 0):
								B = get_patch(nx, ny, filter_size*2 + 1)
								a = img[:, nx:nx+lenx, ny:ny+leny] - img[:, A2[0]:A2[1], A2[2]:A2[3]]
								a = (a * area)**2
								a = (CNT[B[0]:B[1], B[2]:B[3]].sum() +  a.sum() + (abs(x - (nx + lenx // 2)) + abs(y - (ny + leny // 2))) * Nor)
								if(a < minn):
									minn = a
									fx = nx
									fy = ny	
	
	lim = min(mask.shape[0], mask.shape[1])
	for t in range(8):
		dx = DX[t]
		dy = DY[t]
		nx = int(A2[0] + dx)
		ny = int(A2[2] + dy)
		cnt = 0
		while(cnt * 6 < lim and nx>=0 and nx < mask.shape[0]-lenx and ny>=0 and ny<=mask.shape[1]-leny):
			++cnt
			if (or_mask[nx:nx+lenx,ny:ny+leny].sum() == 0):
				B = get_patch(nx,ny,filter_size*2 + 1)
				a = img[:, nx:nx+lenx, ny:ny+leny] - img[:, A2[0]:A2[1], A2[2]:A2[3]]
				a = (a * area)**2
				a = (CNT[B[0]:B[1], B[2]:B[3]].sum()  + a.sum() + (abs(x - (nx + lenx // 2)) + abs(y - (ny + leny // 2)))* Nor)
				if(a < minn):
					minn = a
					fx = nx
					fy = ny
			nx += dx
			ny += dy
	A = get_patch(x, y, filter_size * 4)
	if(or_mask[A[0]:A[1], A[2]:A[3]].any() == 0):
		for i in range(A[0], A[1] - lenx):
			for j in range(A[2], A[3] - leny):
				if(or_mask[i:i+lenx, j:j+leny].sum() == 0):
					B = get_patch(i, j, filter_size*2 + 1)
					a = img[:, i:i+lenx, j:j+leny] - img[:, A2[0]:A2[1], A2[2]:A2[3]]
					a = (a * area)**2
					a = (CNT[B[0]:B[1], B[2]:B[3]].sum() + a.sum() + (abs(x - (i + lenx // 2)) + abs(y - (j + leny // 2)))* Nor)
					if(a < minn):
						minn = a
						fx = i
						fy = j
	d = mask.shape[0] // 15
	for i in range(0, mask.shape[0] - lenx, d):
		for j in  range(0, mask.shape[1] - leny, d):
			if(or_mask[i:i+lenx, j:j+leny].sum() == 0):
				B = get_patch(i, j, filter_size*2 + 1)
				a = img[:, i:i+lenx, j:j+leny] - img[:, A2[0]:A2[1], A2[2]:A2[3]]
				a = (a * area)**2
				a = (CNT[B[0]:B[1], B[2]:B[3]].sum() + a.sum()  + (abs(x - (i + lenx // 2)) + abs(y - (j + leny // 2)))* Nor)
				if(a < minn):
					minn = a
					fx = i
					fy = j
	A = get_patch(fx, fy, filter_size-2)
	if(CNT[A[0]:A[1], A[2]:A[3]].sum() >= 2):
		fx, fy = limited_strentch(fx, fy, x, y, filter_size)
	CNT[fx][fy] += 1
	return tuple([fx,fy])
				
def copy_pixel(sourse_lx, sourse_ly, masked_x, masked_y, filter_size): #目标区域的左上角和当前区域的点p
	A2 = get_patch(masked_x, masked_y, filter_size)
	A1 = [sourse_lx, sourse_lx + A2[1] - A2[0], sourse_ly, sourse_ly + A2[3] - A2[2]]
	
	area = mask[A2[0]:A2[1], A2[2]:A2[3]]
	
	tmp = area * img[:,A1[0]:A1[1], A1[2]:A1[3]] + (1 - area) * (img[:, A2[0]:A2[1], A2[2]:A2[3]])
	img[:, A2[0]:A2[1], A2[2]:A2[3]] = tmp #担心重叠区域，不清楚numpy乘法的实现机制
	
	tmp = area * origin[:,A1[0]:A1[1], A1[2]:A1[3]] + (1 - area) * (origin[:, A2[0]:A2[1], A2[2]:A2[3]])
	origin[:, A2[0]:A2[1], A2[2]:A2[3]] = tmp
	
	tmp = area * img_gray[A1[0]:A1[1], A1[2]:A1[3]] + (1 - area) * (img_gray[A2[0]:A2[1], A2[2]:A2[3]])
	img_gray[A2[0]:A2[1], A2[2]:A2[3]] = tmp
	
	tmp = area * from_x[A1[0]:A1[1], A1[2]:A1[3]] + (1 - area) * from_x[A2[0]:A2[1], A2[2]:A2[3]]
	from_x[A2[0]:A2[1], A2[2]:A2[3]] = tmp
	
	tmp = area * from_y[A1[0]:A1[1], A1[2]:A1[3]] + (1 - area) * from_y[A2[0]:A2[1], A2[2]:A2[3]]
	from_y[A2[0]:A2[1], A2[2]:A2[3]] = tmp
	
	mask[A2[0]:A2[1], A2[2]:A2[3]] -= area
	
'''remove 与 add的调用应该保持一致'''
def remove(x, y, filter_size):
	A = get_patch(x, y, filter_size)
	for i in range(A[0], A[1]):
		for j in range(A[2], A[3]):
			if(bound[i][j] == 1):
				bound_point.pop(i * mask.shape[1] + j)

def change_bound(x, y, filter_size): #filter_size比原来更大,为奇数
	A2 = get_patch(x, y, filter_size)
	tmpmask = mask[A2[0]:A2[1], A2[2]:A2[3]]
	A = [0, 0, 0, 0] #A为实际更改的地方,与计算区域的偏差
	if (x - A2[0] == filter_size // 2):
		A[0] += 1
	if (A2[1] - x - 1== filter_size // 2):
		A[1] -= 1
	if (y - A2[2] == filter_size // 2):
		A[2] += 1
	if (A2[3] - y - 1== filter_size // 2):
		A[3] -= 1
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
	dilated = cv2.dilate(tmpmask,kernel)     #膨胀图像
	tmpbound = dilated - tmpmask
	tmp_scharr_bound_x = cv2.Scharr(tmpmask, cv2.CV_64F, dx=1, dy = 0)
	tmp_scharr_bound_y = cv2.Scharr(tmpmask, cv2.CV_64F, dx=0, dy = 1)
	A2 = A2 + A
	bound[A2[0]:A2[1], A2[2]:A2[3]] = tmpbound[A[0]:tmpmask.shape[0]+A[1], A[2]:tmpmask.shape[1]+A[3]]
	scharr_bound_x[A2[0]:A2[1], A2[2]:A2[3]] = tmp_scharr_bound_x[A[0]:tmpmask.shape[0]+A[1], A[2]:tmpmask.shape[1]+A[3]]
	scharr_bound_y[A2[0]:A2[1], A2[2]:A2[3]] = tmp_scharr_bound_y[A[0]:tmpmask.shape[0]+A[1], A[2]:tmpmask.shape[1]+A[3]]
	
def change_grad(x, y, filter_size): #filter_size比原来更大，为奇数, 注意有下边界
	A2 = get_patch(x, y, filter_size)
	tmp_gray = img_gray[A2[0]:A2[1], A2[2]:A2[3]]
	A = [0, 0, 0, 0]
	if (x - A2[0] == filter_size // 2):
		A[0] += 1
	if (A2[1] - x - 1== filter_size // 2):
		A[1] -= 1
	if (y - A2[2] == filter_size // 2):
		A[2] += 1
	if (A2[3] - y - 1== filter_size // 2):
		A[3] -= 1
	tmp_scharrx = cv2.Scharr(tmp_gray, cv2.CV_64F, dx=1, dy=0) #有正负
	tmp_scharry = cv2.Scharr(tmp_gray, cv2.CV_64F, dx=0, dy=1)
	A2 = A2 + A
	scharrx[A2[0]:A2[1], A2[2]:A2[3]] = tmp_scharrx[A[0]:tmp_gray.shape[0]+A[1], A[2]:tmp_gray.shape[1]+A[3]]
	scharry[A2[0]:A2[1], A2[2]:A2[3]] = tmp_scharry[A[0]:tmp_gray.shape[0]+A[1], A[2]:tmp_gray.shape[1]+A[3]]
	
def getpriority(x, y, filter_size):
	A = get_patch(x, y, filter_size)
	tmp = trust[A[0]:A[1], A[2]:A[3]].mean()
	A = get_patch(x, y, 3)
	area = (1 - bound[A[0]:A[1], A[2]:A[3]]) * (1 - mask[A[0]:A[1], A[2]:A[3]]) #非bound且非mask
	tmpx = 0
	tmpy = 0
	if (area.sum() != 0):
		tmpx = (area * scharrx[A[0]:A[1], A[2]:A[3]]).sum() / area.sum()
		tmpy = (area * scharry[A[0]:A[1], A[2]:A[3]]).sum() / area.sum()
	return tmp * (abs(scharr_bound_x[x][y] * tmpy - scharr_bound_y[x][y] * tmpx) + 0.4) / 255. ##加上0.1是为了应对一些纯色图

def add(x, y, filter_size):
	A = get_patch(x, y, filter_size)
	for i in range(A[0], A[1]):
		for j in range(A[2], A[3]):
			if(bound[i][j] == 1):
				bound_point.update({i * mask.shape[1] + j: getpriority(i, j, filter_size)})
				
def change_trust(x, y, filter_size):	
	A = get_patch(x, y, filter_size)
	tmp = trust[A[0]:A[1], A[2]:A[3]].mean()
	for i in range(A[0], A[1]):
		for j in range(A[2], A[3]):
			if(trust[i][j] == 0):
				trust[i][j] = tmp
	
def main_pic(filter_size):
	global mask
	global img #LAB
	global img_gray
	global bound
	global scharrx
	global scharry
	global scharr_bound_x
	global scharr_bound_y
	global bound_point
	global trust #信任度
	global origin
	global or_mask
	global from_x
	global from_y
	global DX
	global DY
	global CNT
	DX = [0,0,1,-1,1,1,-1,-1,0]
	DY = [1,-1,0,0,1,-1,1,-1,0]
	
	bound_point = {}
	if (filter_size % 2 != 1):
		print("filter_size is not odd number!")
		return 0
		
	mask = cv2.imread("mask.png")	
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) #一维
	mask = mask.astype(np.float32) / 255.0
	mask = np.round(mask)#hole为1
	trust = 1 - mask
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
	dilated = cv2.dilate(mask,kernel)      #膨胀图像
	bound = dilated - mask
	origin = cv2.imread("gt.png")
	for_out_gt = origin.copy()
	origin = origin.astype(np.float32)
	if(mask.shape[0] != origin.shape[0] or mask.shape[1] != origin.shape[1]):
		print("shape match missed")
		return 0
	from_x = np.zeros(mask.shape)
	from_y = np.zeros(mask.shape)
	for i in range(mask.shape[0]):
		from_x[i] = i
	for j in range(mask.shape[1]):
		from_y[:,j] = j
	CNT = np.zeros((mask.shape[0], mask.shape[1]))
	or_mask = mask.copy()
	origin[:,:,0] *= 1 - mask
	origin[:,:,1] *= 1 - mask
	origin[:,:,2] *= 1 - mask
	for_out_masked = origin.copy()
	img = origin.copy()
	cv2.imwrite("bemasked.png", origin)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	img = img.transpose(2, 0, 1)
	origin = origin.transpose(2, 0, 1)
	scharrx = cv2.Scharr(img_gray, cv2.CV_64F, dx=1, dy=0) #有正负
	scharry = cv2.Scharr(img_gray, cv2.CV_64F, dx=0, dy=1)
	scharr_bound_x = cv2.Scharr(mask, cv2.CV_64F, dx=1, dy = 0)
	scharr_bound_y = cv2.Scharr(mask, cv2.CV_64F, dx=0, dy = 1)
	for i in range(mask.shape[0]): #用pop删除元素，update新增元素
		for j in range(mask.shape[1]):
			if(bound[i][j] == 1):
				bound_point.update({i * mask.shape[1] + j: getpriority(i, j, filter_size)})
	cnt = 0
	while len(bound_point) > 0:
		maxx = -1
		point = 0
		print(mask.sum())
		for i in bound_point:
			if (bound_point[i] > maxx):
				maxx = bound_point[i]
				point = i
		cnt += 1
		if(cnt % 20 == 0):
			origin = origin.transpose(1, 2, 0)
			cv2.imwrite("pp.png", origin)
			origin = origin.transpose(2, 0, 1)
			
		x = point // mask.shape[1]
		y = point % mask.shape[1]
		sourse_p = find_nearst(x, y, filter_size)
		i = sourse_p[0]
		j = sourse_p[1]
		remove(x, y, filter_size * 2 + 1)
		copy_pixel(sourse_p[0], sourse_p[1], x, y, filter_size)
		change_bound(x, y, filter_size + 4)
		change_grad(x, y, filter_size + 4)
		change_trust(x, y, filter_size) #必须在add之前
		add(x, y, filter_size * 2 + 1)
	vv = np.zeros((mask.shape[0], mask.shape[1], 3))
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if(CNT[i][j] > 0):
				vv[i][j] = 255
	origin = origin.transpose(1, 2, 0)
	
	Output = np.zeros((mask.shape[0] * 2 + 9, mask.shape[1]*2 + 9, 3))
	Output[:mask.shape[0],:mask.shape[1],:] =  for_out_masked
	Output[:mask.shape[0],mask.shape[1]+9:,:] = for_out_gt
	Output[mask.shape[0] + 9:,:mask.shape[1],:] = origin
	Output[mask.shape[0] + 9:,mask.shape[1]+9:,:] = vv
	cv2.imwrite("final.png", Output)
	cv2.imwrite("predict.png", origin)

main_pic(9)
