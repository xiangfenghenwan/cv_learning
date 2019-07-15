#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import random
import numpy as np
from matplotlib import pyplot as plt


# In[6]:


#灰度图像读取和显示
img_gray = cv2.imread("E:/graduate/CVtraining/lesson 1/lenna.jpg", 0) #路径好像不能有中文，0：通道
cv2.imshow("lenna", img_gray)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[7]:


#显示图片灰度值矩阵
print(img_gray)


# In[8]:


#显示图片数据类型
print(img_gray.dtype)


# In[9]:


#显示灰度图形状，高度h，宽度w
print(img_gray.shape)


# In[1]:


#彩色图像读取显示
import cv2
img = cv2.imread("E:/graduate/CVtraining/lesson 1/tr.jpg")
cv2.imshow("tr", img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[2]:


print(img)
print(img.shape)


# In[5]:


#image crop 图像裁剪
img_crop = img[300:900, 400:1250]
cv2.imshow("img_crop",img_crop)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[6]:


print(img_crop)
print(img_crop.shape)


# In[8]:


#颜色分割 color split
B, G, R = cv2.split(img_crop)
cv2.imshow("B", B)
cv2.imshow("G", G)
cv2.imshow("R", R)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[13]:


#改变颜色 change color
import random
def random_light_color(img):
    B, G, R = cv2.split(img_crop)
    
    #b改变
    b_rand = random.randint(-50,50) #随机产生一个-50到50的整数
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B>lim] = 255
        B[B<=lim] = (b_rand + B[B<=lim]).astype(img_crop.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B<lim] = 255
        B[B>=lim] = (b_rand + B[B>=lim]).astype(img_crop.dtype)
    
    #g改变    
    g_rand = random.randint(-50, 50)
    if b_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G>lim] = 255
        G[G<=lim] = (g_rand + G[G<=lim]).astype(img_crop.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G<lim] = 255
        G[G>=lim] = (g_rand + G[G>=lim]).astype(img_crop.dtype)
    
    #r改变
    r_rand = random.randint(-50, 50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img_crop.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img_crop.dtype)
        
    img_merge = cv2.merge((B,G,R)) #合并改变后的颜色  双括号
    return img_merge

img_random_color = random_light_color(img_crop)
cv2.imshow("img_random_color", img_random_color)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[14]:


#图像变换


# In[17]:


#伽马校正  gamma correction

img_dark = cv2.imread("E:/graduate/CVtraining/lesson 1/dark.jpg")
cv2.imshow("img_dark", img_dark)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

import numpy as np
def adjust_gamma(image,gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i/255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(img_dark,table)   #look up table
img_brighter = adjust_gamma(img_dark, 2)
cv2.imshow("img_dark", img_dark)
cv2.imshow("img_brighter", img_brighter)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[19]:


#直方图 histogram
from matplotlib import pyplot as plt
img_small_brighter = cv2.resize(img_brighter, (int(img_brighter.shape[0]*0.5), int(img_brighter.shape[1]*0.5)))
plt.hist(img_brighter.flatten(), 256, [0, 256], color = 'r')
img_yuv = cv2.cvtColor(img_small_brighter, cv2.COLOR_BGR2YUV)
# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])   # only for 1 channel
# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)   # y: luminance(明亮度), u&v: 色度饱和度
cv2.imshow('Color input image', img_small_brighter)
cv2.imshow('Histogram equalized', img_output)
key = cv2.waitKey(0)
if key == 27:
    exit()


# In[22]:


#图像旋转  rotation
M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 1) # center, angle, scale
img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0])) #将M作用到img上
cv2.imshow('rotated lenna', img_rotate)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()
print(M)

# set M[0][2] = M[1][2] = 0
print(M)
img_rotate2 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('rotated lenna2', img_rotate2)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()
# explain translation

# scale+rotation+translation = similarity transform
M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 0.5) # center, angle, scale
img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('rotated lenna', img_rotate)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()

print(M)    


# In[23]:


#仿射变换 Affine Transform
rows, cols, ch = img.shape
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])
 
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('affine lenna', dst)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()


# In[25]:


# perspective transform
def random_warp(img, row, col):
    height, width, channels = img.shape

    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return M_warp, img_warp
M_warp, img_warp = random_warp(img, img.shape[0], img.shape[1])
cv2.imshow('lenna_warp', img_warp)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()


# In[ ]:




