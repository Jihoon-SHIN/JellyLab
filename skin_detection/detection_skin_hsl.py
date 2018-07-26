# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))));
# from detection import RangeColorDetector
# import detection.py
from detection_hsl import RangeColorDetector
import numpy as np
import cv2

#Firs image boundaries
min_range1 = np.array([0, 50, 50], dtype = "uint8") #lower HSV boundary of skin color
max_range1 = np.array([20, 255, 255], dtype = "uint8") #upper HSV boundary of skin color
min_range2 = np.array([165, 50, 50], dtype="uint8")
max_range2 = np.array([179, 255, 255], dtype ="uint8")
my_skin_detector = RangeColorDetector(min_range1, max_range1, min_range2, max_range2) #Define the detector object
image = cv2.imread("dd.jpg") #Read the image with OpenCV
#We do not need to remove noise from this image so morph_opening and blur are se to False
image_filtered = my_skin_detector.returnFiltered(image, morph_opening=True, blur=True, kernel_size=3, iterations=1)
cv2.imwrite("25_hsl-6.jpg", image_filtered) #Save the filtered image



