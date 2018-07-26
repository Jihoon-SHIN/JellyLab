import numpy as np
import cv2
import sys

class RangeColorDetector:

    def __init__(self):
        self.a = 1;
    def setRange(self):
        self.a = 2;
    def getRange(self):
        return self.a

    def returnFiltered(self, frame, morph_opening=True, blur=True, kernel_size=5, iterations=1):
        frame_filtered = self.returnMask(frame, morph_opening=morph_opening, blur=blur, kernel_size=kernel_size, iterations=iterations)
        frame_denoised = cv2.cvtColor(frame_filtered, cv2.COLOR_HLS2BGR)
        return frame_denoised

    def returnMask(self, frame, morph_opening=True, blur=True, kernel_size=5, iterations=1):
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        hsv_clone = frame_hsv
        rows, cols, num = frame_hsv.shape
        frame_size = rows*cols
        red_count = 0
        skin_count = 0

        check_skin =0;
        check_atopy =0;
        for i in range(rows):
            for j in range(cols):
                H = frame_hsv[i, j, 0]
                L = frame_hsv[i, j, 1]
                S = frame_hsv[i, j, 2]
                if(S==0):
                    LS_ration = 1
                else:
                    LS_ratio = L*1.0/S
                is_skin = (S>=10) and (LS_ratio>0.5) and (LS_ratio<5.0) and ((H<=14) or (H>=165))
                if (is_skin==0):
                    hsv_clone[i,j,0] = 0
                    hsv_clone[i,j,1] = 0
                    hsv_clone[i,j,2] = 0
                else:
                    skin_count = skin_count + 1
                if (is_skin==1 and (H<=7) or (H>=173)):
                    red_count = red_count + 1

        if(skin_count !=0):
            check_skin = (skin_count*1.0)/(frame_size)
        if(red_count !=0):
            check_atopy = (red_count*1.0)/(skin_count)
            print(red_count)
            print(skin_count)
            print(check_atopy)

        if(check_skin < 0.3):
            print("It is not skin")
        if(check_atopy > 0.3):
            print("It is bad status")

        return hsv_clone