
from pylab import *

img = imread('chick.png')
(rows, cols, d3) = img.shape

figure()
imshow(img)

############################################
# First effect (ghost chick):

img1 = array(img) # copy img as img1

for i in range(rows):
    for j in range(cols):
        for k in range(d3):
            img1[i, j, k] = 1 - img1[i, j, k]

figure()
imshow(img1)

############################################
# Second effect (horror chick):

img2 = array(img) # copy img as img2

for i in range(rows):
    for j in range(cols):
        if sum(img2[i, j]) < 1.5:
            img2[i, j] = (.0, .0, .0)

figure()
imshow(img2)

############################################
# Finally, show all the images
show()

