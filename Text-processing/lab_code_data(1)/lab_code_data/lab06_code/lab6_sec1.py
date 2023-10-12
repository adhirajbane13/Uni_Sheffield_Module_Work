from pylab import *
img = imread('C:\\UoS_DA_Lab\\My_Work\\Text-processing\\lab_code_data(1)\\lab_code_data\\lab06_code\\chick.png')
#imshow(img)
img1 = img

(row,col,d) = img.shape

for i in range(row):
    for j in range(col):
        for k in range(d):
            img1[i,j,k] = 1 - img[i,j,k]

for i in range(row):
    for j in range(col):
        pixel = img[i,j]
        if sum(pixel) < 1.5:
            img[i,j] = (0.0,0.0,0.0)

imshow(img1)
show()