# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 14:06:10 2018

@author: Okale
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as tr

def falsos(imagen_1,imagen_2,f = "default"):
    falsos = 0
    filas = 220
    columnas = 360
    for i in range(filas):
        for j in range(columnas):
            if(f == "positivos"):
                if(imagen_1[i][j] == 0 and imagen_2[i][j] == 1):
                    falsos += 1
            elif(f == "negativos"):
                if(imagen_2[i][j] == 0 and imagen_1[i][j] == 1):
                    falsos += 1
            else:
                print("You have to assign an opcion betwen 'positivos' and 'negativos'.")
    return falsos
   
def exactitud(imagen_1,imagen_2,ex = "default"):
    filas = 220
    columnas = 360
    contador_parcial = 0
    contador_total = 0
    
    for i in range(filas):
        for j in range(columnas):
            if(ex == "positivos"):
                if(imagen_1[i][j] == 1 and imagen_2[i][j] == 1):
                    contador_parcial += 1
                    contador_total += 1
                elif(imagen_1[i][j] == 1):
                    contador_total += 1
            elif(ex == "negativos"):
                if(imagen_1[i][j] == 0 and imagen_2[i][j] == 0):
                    contador_parcial += 1
                    contador_total += 1
                elif(imagen_1[i][j] == 0):
                    contador_total += 1
            else:
                print("You have to assign an opcion betwen 'positivos' and 'negativos'.")
    return contador_total,contador_parcial


if __name__ == "__main__":   

    exactitud_pos_promedio = 0
    exactitud_neg_promedio = 0
    tabla = np.zeros((23,6),dtype = "float64")
    
    for i in range(23):
        indice = i + 1
        image_1 = cv.imread("imagenesParaSegmentar/segmentadas_manual/%d.jpg"%indice,cv.IMREAD_GRAYSCALE)
        image_2 = cv.imread("imagenesParaSegmentar/segmentadas_programa/%d.jpg"%indice,cv.IMREAD_GRAYSCALE)
        
        
        
        image_1 = image_1[30:250,85:445]
        image_2 = image_2[40:247,53:390]
        image_2 = tr.resize(image_2,(220,360),mode = 'constant')
        
        plt.imshow(image_1,cmap="gray")
        plt.show()
        plt.imshow(image_2,cmap="gray")
        plt.show()
        
        image_1 = np.array(image_1, dtype="bool")
        image_2 = np.array(image_2, dtype="bool")
        
        ex_total_pos,ex_parcial_pos = exactitud(image_1,image_2,ex = "positivos")
        ex_total_neg,ex_parcial_neg = exactitud(image_1,image_2,ex = "negativos") 
        falsos_positivos = falsos(image_1,image_2,f = "positivos")
        falsos_negativos = falsos(image_1,image_2,f = "negativos")
        
        exactitud_pos_promedio += ex_parcial_pos
        exactitud_neg_promedio += ex_parcial_neg
        
        n = ex_parcial_pos + ex_parcial_neg + falsos_positivos + falsos_negativos
        
        tabla[i,0] = indice
        tabla[i,1] = ex_parcial_pos / (ex_parcial_pos + falsos_negativos)
        tabla[i,2] = ex_parcial_pos / (ex_parcial_pos + falsos_positivos)
        tabla[i,3] = falsos_positivos / (falsos_positivos + ex_parcial_neg)
        tabla[i,4] = (ex_parcial_pos + ex_parcial_neg) / n
        tabla[i,5] = (falsos_positivos + falsos_negativos) / n
        
        print("exactitud 1: {expp}/{extp} \n"
              "exactitud 0: {expn}/{extn} \n"
              "falsos_positivos: {fp} \n"
              "falsos_negativos: {fn} \n".format(expp = ex_parcial_pos,
              extp = ex_total_pos,expn = ex_parcial_neg,extn = ex_total_neg,
              fp = falsos_positivos, fn = falsos_negativos))