
from pylab import *

img = imread('che.png')
(rows, cols, d3) = img.shape

figure()
imshow(img)

############################################
# First effect (strongly monochrome):

img1 = array(img) # copy image array

for i in range(rows):
    for j in range(cols):
        for k in range(d3):
            if img1[i, j, k] < 0.5:
                img1[i, j, k] = 0.0
            else:
                img1[i, j, k] = 1.0

figure()
imshow(img1)

##############################################
# Second effect (red/black):

img2 = array(img) # copy image array

for i in range(rows):
    for j in range(cols):
        if img2[i, j, 0] < 0.5:
            img2[i, j] = (.0, .0, .0)
        else:
            img2[i, j] = (1., .0, .0)

figure()
imshow(img2)

##############################################
# Third effect (face white, background red):

img3 = array(img) # copy image array

for i in range(rows):
    for j in range(cols):
        if img3[i, j, 0] < 0.5:
            img3[i, j] = (.0, .0, .0) # black
        elif 55 < i < 160 and 55 < j < 140:
            img3[i, j] = (1., 1., 1.) # white
        else:
            img3[i, j] = (1., .0, .0) # red

figure()
imshow(img3)

############################################
# Fourth effect (psychedelic):

img4 = array(img) # copy image array

for i in range(rows):
    for j in range(cols):
        if img4[i, j, 0] > 0.66:
            img4[i, j] = (1., .0, .0) # red
        elif img4[i, j, 0] < 0.33:
            img4[i, j] = (.0, .0, 1.) # blue
        else:
            img4[i, j] = (.0, 1., .0) # green

figure()
imshow(img4)

############################################
# Fifth effect (overlay chick image):

img5 = array(img) # copy image array
chick = imread('chick.png')
(rows, cols, d3) = chick.shape

(dx, dy) = (60, 58) # displacement of chick image from top left
                  # corner, as components in x, y directions

for i in range(rows):
    for j in range(cols):
        img5[i+dy, j+dx] = chick[i, j]

figure()
imshow(img5)

############################################
# Finally, show all the images
show()

