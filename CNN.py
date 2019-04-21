# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 23:30:13 2019

@author: Okale
"""
from xml.dom import minidom as dom
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from pickle import load

imagen_1 = np.zeros((220,360))
imagen_2 = np.zeros((220,360))
#ruta = "Autos/segmentacion/boundingboxes/%d.xml"
#ruta_d = "Autos/segmentacion/dict_autos/mobilenetV2_boxes"
#ruta_d = "Autos/segmentacion/dict_autos/mobilenetV1_boxes"
#ruta_d = "Autos/segmentacion/dict_autos/faster_rcnn_inception_boxes"

ruta = "Personas/segmentacion/boundingboxes/%d.xml"
#ruta_d = "Personas/segmentacion/dict_personas/mobilenetV2_boxes"
ruta_d = "Personas/segmentacion/dict_personas/mobilenetV1_boxes"
#ruta_d = "Personas/segmentacion/dict_personas/faster_rcnn_inception_boxes"
indice = 0
num_img = 100
iou = []
exactitud = 0
with open(ruta_d,"rb") as f:
    b_boxes = load(f)

for i in range(num_img):
    indice +=1
    detects = b_boxes[indice]
    doc = dom.parse(ruta%indice)
    #path = doc.getElementsByTagName("path")[0]
    #path = path.firstChild.data
    objects = doc.getElementsByTagName("object")
    for obj in objects:
        name = obj.getElementsByTagName("name")[0]
        name = name.firstChild.data
        bnd = obj.getElementsByTagName("bndbox")[0]
        ymin = bnd.getElementsByTagName("xmin")[0]
        ymin = int(ymin.firstChild.data)
        ymax = bnd.getElementsByTagName("xmax")[0]
        ymax = int(ymax.firstChild.data)
        xmin = bnd.getElementsByTagName("ymin")[0]
        xmin = int(xmin.firstChild.data)
        xmax = bnd.getElementsByTagName("ymax")[0]
        xmax = int(xmax.firstChild.data)
        #print("nombre: {n}\n xmin: {minx}\nxmax:{maxx}".format(n=name,minx=xmin,maxx=xmax))
        imagen_1[xmin:xmax,ymin:ymax] = True        
        imagen_1 = imagen_1 > 0
    if(len(detects) > 0):
        for detect in detects:
            xmin = int(round(detects[detect]['box'][0] * 220))
            ymin = int(round(detects[detect]['box'][1] * 360))
            xmax = int(round(detects[detect]['box'][2] * 220))
            ymax = int(round(detects[detect]['box'][3] * 360))
            imagen_2[xmin:xmax,ymin:ymax] = True
            imagen_2 = imagen_2 > 0
    else:
        imagen_2 = imagen_2 > 0
        
    
    im_and = np.bitwise_and(imagen_1,imagen_2)
    im_or = np.bitwise_or(imagen_1,imagen_2)
    iou.append(np.sum(im_and) / np.sum(im_or))
    if((iou[i] * 100) > 50):
        exactitud += 1
    """
    plt.imshow(imagen_1,cmap='gray')
    plt.show()    
    plt.imshow(imagen_2,cmap='gray')
    plt.show()
    """
    imagen_1[:,:] = 0
    imagen_2[:,:] = 0

iou_prom = np.sum(iou) / 100
exactitud = np.sum(exactitud)


"""
imagen_2[80:150,100:180] = True
imagen_2 = imagen_2 > 0
overlap = np.bitwise_and(imagen_1,imagen_2)
union = imagen_1 != imagen_2
iou = np.sum(overlap) / np.sum(union)
#im<agen_1 = imagen_1 > 255
#imagen_2 = imagen_2 > 255
plt.imshow(imagen_1,cmap='gray')
plt.show()
plt.imshow(imagen_2,cmap='gray')
plt.show()
plt.imshow(overlap,cmap='gray')
plt.show()
plt.imshow(union,cmap='gray')
plt.show()
print(iou)
"""