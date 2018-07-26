# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))));
# from detection import RangeColorDetector
# import detection.py
from detection_a import RangeColorDetector
import numpy as np
import cv2

my_skin_detector = RangeColorDetector() #Define the detector object
image = cv2.imread("25.jpg") #Read the image with OpenCV
#We do not need to remove noise from this image so morph_opening and blur are se to False
image_filtered = my_skin_detector.returnFiltered(image, morph_opening=True, blur=True, kernel_size=3, iterations=1)
cv2.imwrite("25-5.jpg", image_filtered) #Save the filtered image

