import cv2
import numpy as np

image = cv2.imread(r"C:\Users\tamil\OneDrive\Documents\Computer Vision\jellyfish-memory-learning-neuroscience.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    extracted_object = image[y:y + h, x:x + w]
    cv2.imshow('Extracted Object', extracted_object)
    cv2.waitKey(0)
cv2.imshow('Objects with Rectangles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



