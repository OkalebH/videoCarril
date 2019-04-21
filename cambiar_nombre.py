# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 02:28:48 2018

@author: Okale
"""

import cv2 as cv
#from os import listdir
from os.path import isfile
cont = 0

for i in range(271,371):
    
    if(isfile("Personas/si_personas/%d.jpg"%i)):
        cont += 1
        img = cv.imread("Personas/si_personas/%d.jpg"%i,cv.IMREAD_GRAYSCALE)
        cv.imwrite("Personas/segmentacion/imagenes/%d.jpg"%cont,img)
        cv.destroyAllWindows()
    else:
        continue
cont=0
for i in range(171,271):
    
    if(isfile("Autos/si_autos/%d.jpg"%i)):
        cont += 1
        img = cv.imread("Autos/si_autos/%d.jpg"%i,cv.IMREAD_GRAYSCALE)
        cv.imwrite("Autos/segmentacion/imagenes/%d.jpg"%cont,img)
        cv.destroyAllWindows()
    else:
        continue