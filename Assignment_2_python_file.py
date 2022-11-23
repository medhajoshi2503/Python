# Import opencv
import cv2
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

import imageio as iio
 
from scipy.ndimage.filters import generic_filter
##from scipy.ndimage import imread


# Use the second argument or (flag value) zero
# that specifies the image is to be read in grayscale mode
##clown image is added in google collab
image = cv2.imread('/content/clown.jpeg')  ##

##---Question 1 part a
##--Read the image ‘clown.jpeg’ and display it.
from google.colab.patches import cv2_imshow
cv2_imshow(image)  


##---Question 1 part b
##--Convert it into a grayscale image.

# Use the cvtColor() function to grayscale the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.waitKey(0)

cv2_imshow(gray_image)

##c) To suppress the noise use a Gaussian kernel to smoothen it. Keep the
##kernel size as 5 x 5 and sigma = 1.5.
## sigma = 1.5 so kernelsize 2*signma+1 =  2*1.5 + 1  =44

kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(gray_image,-1,kernel)

plt.subplot(121),plt.imshow(gray_image),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('smoothning')
plt.xticks([]), plt.yticks([])
plt.show()



# Gaussian Blurring
blur = cv2.GaussianBlur(gray_image,(5,5),0)
cv2_imshow(blur)








# Apply Sobelx in high output datatype 'float32'
# and then converting back to 8-bit to prevent overflow
sobelx_64 = cv2.Sobel(blur,cv2.CV_32F,1,0,ksize=3)
absx_64 = np.absolute(sobelx_64)
sobelx_8u1 = absx_64/absx_64.max()*255
sobelx_8u = np.uint8(sobelx_8u1)
cv2_imshow(sobelx_8u)

# Similarly for Sobely
sobely_64 = cv2.Sobel(blur,cv2.CV_32F,0,1,ksize=3)
absy_64 = np.absolute(sobely_64)
sobely_8u1 = absy_64/absy_64.max()*255
sobely_8u = np.uint8(sobely_8u1)
cv2_imshow(sobely_8u)

# From gradients calculate the magnitude and changing
# it to 8-bit (Optional)
mag = np.hypot(sobelx_8u, sobely_8u)
mag = mag/mag.max()*255
mag = np.uint8(mag)
###plt.plot(mag)
print("Magnitude is")
print(mag)
# Find the direction and change it to degree
theta = np.arctan2(sobely_64, sobelx_64)
angle = np.rad2deg(theta)
print("Angle is")
print(angle)
##n.plot(angle)



##nMS

# Find the neighbouring pixels (b,c) in the rounded gradient direction
# and then apply non-max suppression
M, N = mag.shape
Non_max = np.zeros((M,N), dtype= np.uint8)

for i in range(1,M-1):
    for j in range(1,N-1):
       # Horizontal 0
        if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180) or (-22.5 <= angle[i,j] < 0) or (-180 <= angle[i,j] < -157.5):
            b = mag[i, j+1]
            c = mag[i, j-1]
        # Diagonal 45
        elif (22.5 <= angle[i,j] < 67.5) or (-157.5 <= angle[i,j] < -112.5):
            b = mag[i+1, j+1]
            c = mag[i-1, j-1]
        # Vertical 90
        elif (67.5 <= angle[i,j] < 112.5) or (-112.5 <= angle[i,j] < -67.5):
            b = mag[i+1, j]
            c = mag[i-1, j]
        # Diagonal 135
        elif (112.5 <= angle[i,j] < 157.5) or (-67.5 <= angle[i,j] < -22.5):
            b = mag[i+1, j-1]
            c = mag[i-1, j+1]           
            
        # Non-max Suppression
        if (mag[i,j] >= b) and (mag[i,j] >= c):
            Non_max[i,j] = mag[i,j]
        else:
            Non_max[i,j] = 0

