# 非学习方法有更快的版本了哟！！！！
# 最新动态（20190521）网络的输入输出和初步的数据集已经准备好了，将于20190521晚进行调试，希望问题不要太大

# 最近动态（20190519）已经将vgg19加入了网络（由于vgg16调用比较麻烦），并且设定了untrainable，接下来一步是处理输入并且对于网络进行调试（注：Ubuntu下的CUDA和Cudnn我也已经装好了）

# 喜讯(from NaturezzZ) 经过我好几天的不懈努力，终于将cuda和tensorflow-gpu安装成功了！！！

# 喜讯！！网络基本已经搭建好了，基本和论文一致，已经实现pconv，详见newstructure.py。还差一点点，主要是vgg16还没搞上去，所以也还没有运行。可能还有问题，等数据出来再说。
# Inpainting
## Project for Introduction of AI
## We can make it!
## Have a try ^_^ 
## to do:
### 求求考虑一下problems里面的问题
### 数据处理：
#### 将完整图片和蒙版随机匹配，注意通道数（3）和大小需要匹配（512 or 256 ？可以先试一下512吧，但最后参数可能会比较多）
#### 新添加蒙版：均匀地蒙住像素点（为超分辩率做准备，因为只要原来的写成功了，超分辨率就很简单了，所以先添加好蒙版），同时可以对已经有的蒙版做一些随机改变
### 搞清楚VGG应该怎么运用到我们的网络里
