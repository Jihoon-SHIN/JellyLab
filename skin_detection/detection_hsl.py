import numpy as np
import cv2
import sys

class RangeColorDetector:

    def __init__(self, min_range1, max_range1, min_range2, max_range2):
        self.min_range1 = min_range1
        self.max_range1 = max_range1
        self.min_range2 = min_range2
        self.max_range2 = max_range2

    def setRange(self, min_range1, max_range1, min_range2, max_range2):
        self.min_range1 = min_range1
        self.max_range1 = max_range1
        self.min_range2 = min_range2
        self.max_range2 = max_range2

    def getRange(self):
        return (self.min_range1, self.max_range1, self.min_range2, self.max_range2)

    def returnFiltered(self, frame, morph_opening=True, blur=True, kernel_size=5, iterations=1):
        frame_filtered = self.returnMask(frame, morph_opening=morph_opening, blur=blur, kernel_size=kernel_size, iterations=iterations)
        frame_denoised = cv2.cvtColor(frame_filtered, cv2.COLOR_HLS2BGR)
        return frame_denoised

    def returnMask(self, frame, morph_opening=True, blur=True, kernel_size=5, iterations=1):
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        hsv_clone = frame_hsv
        rows, cols, num = frame_hsv.shape
        for i in range(rows):
            for j in range(cols):
                H = frame_hsv[i, j, 0]
                L = frame_hsv[i, j, 1]
                S = frame_hsv[i, j, 2]
                if(S==0):
                    LS_ration = 1
                else:
                    LS_ratio = L/S
                check_skin = (S>=10) and (LS_ratio>0.5) and (LS_ratio<5.0) and ((H<=20) or (H>=165))
                if (check_skin==0):
                    # print(S>=40 , LS_ratio>0.5,LS_ratio<5.0, H<=14, H>=165)
                    hsv_clone[i,j,0] = 0
                    hsv_clone[i,j,1] = 0
                    hsv_clone[i,j,2] = 0
        return hsv_clone