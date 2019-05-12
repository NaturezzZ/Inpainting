# 喜讯(from NaturezzZ) 经过我好几天的不懈努力，终于将cuda和tensorflow-gpu安装成功了！！！
# 喜讯！！现在的MyLayers似乎可以用了，调用方式同layers.Conv2D,快试一下行不行吧！！！！！(初始化只能写成 “name” = val列表的形式)  上面有layers.Conv2D的示例，拜托写个程序试一下MyLayers对不对
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
