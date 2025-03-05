import cv2
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt

#read image
image = cv2.imread("file path here")

#convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#perform binary thresholding
kernel_size = 3

#i chose values between black and darker gray because I think thats the relative color of the hogline, if not we will have to test more
ret,thresh = cv2.threshold(gray, 0, 30, cv2.THRESH_BINARY)

#find contours
contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours = countours[0] if len(contours) == 2 else countours[1]

#draw countours
radius = 2
color = (30, 255, 50)
cv2.drawContours(image, contours, -1, color, radius)

#show the masked image
cv2.imshow(image)



