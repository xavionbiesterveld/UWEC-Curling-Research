import cv2

#read image
image = cv2.imread("file path here")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply CLAHE for better contrast
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
equalized_image = clahe.apply(gray)

# Adaptive thresholding
thresh = cv2.adaptiveThreshold(equalized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 11, 15)

# Morphological operation to enhance the line
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
thresh = cv2.dilate(thresh, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Draw contours
color = (30, 255, 50)
cv2.drawContours(image, contours, -1, color, 2)

# Show images
'''cv2.imshow('img', image)
cv2.imshow('gray', equalized_image)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)'''

cv2.imwrite('curlingimage2_contours.png', image)
cv2.imwrite('curlingimage2_thresh.png', thresh)
cv2.destroyAllWindows()