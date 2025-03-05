import cv2

#read image
image = cv2.imread("curlingimage1.png")

#convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization
equalized_image = cv2.equalizeHist(gray)

#perform binary thresholding
#i chose values between black and darker gray because I think thats the relative color of the hogline, if not we will have to test more
ret,thresh = cv2.threshold(equalized_image, 165, 255, cv2.THRESH_BINARY_INV)

#find contours
contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours = contours[0] if len(contours) == 2 else contours[1]

#draw countours
radius = 2
color = (30, 255, 50)
cv2.drawContours(image, contours, -1, color, radius)

#show the masked image
cv2.imshow('img', image)
cv2.imshow('gray', gray)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)
