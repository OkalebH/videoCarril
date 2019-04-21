# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 15:28:17 2018

@author: Okale
"""

import cv2 as cv
#import time
#import matplotlib as plt

cap = cv.VideoCapture("Videos/Personas.mp4")
cont_no = 0
cont_si = 0
nombre_frame = "Carretera"
while(cap.isOpened()):
    ret,frame = cap.read()
    if (not ret):
        break
    gray = cv.cvtColor(frame,cv.COLOR_RGBA2GRAY)
    imagen_recortada = gray[200:430,500:850]
    cv.imshow(nombre_frame,imagen_recortada)
    #time.sleep(.20)
    if (cv.waitKey(0) & 0xff == ord("q")):
        break
    elif (cv.waitKey(0) & 0xff == ord("c")):
        cont_si += 1
        cv.imwrite("Personas/si_personas/%d.jpg"%cont_si,imagen_recortada)
    elif (cv.waitKey(0) & 0xff == ord("v")):
        cont_no += 1
        cv.imwrite("Personas/no_personas/%d.jpg"%cont_no,imagen_recortada)
    #time.sleep(.20)
cap.release()
cv.destroyAllWindows()