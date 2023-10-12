from pylab import *
img = imread('C:\\UoS_DA_Lab\\My_Work\\Text-processing\\lab_code_data(1)\\lab_code_data\\lab06_code\\che.png')
imga = imread('C:\\UoS_DA_Lab\\My_Work\\Text-processing\\lab_code_data(1)\\lab_code_data\\lab06_code\\chick.png')


img1 = img.copy()
img2 = img.copy()
img3 = img.copy()
img4 = img.copy()
img5 = img.copy()


(row,col,d) = img.shape
(row1,col1,d1) = imga.shape

for i in range(row):
    for j in range(col):
        for k in range(d):
            if img1[i,j,k]< 0.5:
                img1[i,j,k] = 0.0
            else:
                img1[i,j,k] = 1.0



for i in range(row):
    for j in range(col):
        pixel = img2[i,j]
        if sum(pixel) < 1.5:
            img2[i,j] = (0.0,0.0,0.0)
        else:
            img2[i,j] = (1.0,0.0,0.0)



for i in range(row):
    for j in range(col):
        if i>=60 and i<=158 and j>=57 and j<=132:
            img3[i,j] = (1.0,1.0,1.0)
        else:
            pixel = img3[i,j]
            if sum(pixel) < 1.5:
                img3[i,j] = (0.0,0.0,0.0)
            else:
                img3[i,j] = (1.0,0.0,0.0)

figure()
imshow(img3)

for i in range(row):
    for j in range(col):
        pixel = img4[i,j]
        if sum(pixel) > 0.66:
            img4[i,j] = (1.0,0.0,0.0)
        elif sum(pixel) < 0.33:
            img4[i,j] = (0.0,0.0,1.0)
        else:
            img4[i,j] = (0.0,1.0,0.0)

figure()
imshow(img4)

for i in range(row1):
    for j in range(col1):
        img5[i+60,j+57] = imga[i,j]

figure()
imshow(img5)
show()