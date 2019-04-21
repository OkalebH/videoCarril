# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:31:26 2019

@author: Okale
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as tr
from skimage.filters import threshold_otsu as otsu

def falsos_PN(imagen_1,imagen_2,imagen_3,f = "default"): 
    falsos_1 = 0
    falsos_2 = 0
    filas = 220
    columnas = 360
    for i in range(filas):
        for j in range(columnas):
            if(f == "positivos"):
                if(imagen_1[i][j] == 0 and imagen_2[i][j] == 1):
                    falsos_1 += 1
                if(imagen_1[i][j] == 0 and imagen_3[i][j] == 1):
                    falsos_2 += 1
            elif(f == "negativos"):
                if(imagen_2[i][j] == 0 and imagen_1[i][j] == 1):
                    falsos_1 += 1
                if(imagen_3[i][j] == 0 and imagen_1[i][j] == 1):
                    falsos_2 += 1
            else:
                print("You have to assign an opcion between 'positivos' and 'negativos'.")
    return falsos_1,falsos_2
    
def verdaderos_PN(imagen_1,imagen_2,imagen_3,ex = "default"):
    filas = 220
    columnas = 360
    verdaderos_pn_1 = 0
    verdaderos_pn_2 = 0
    pn_totales = 0
    
    for i in range(filas):
        for j in range(columnas):
            if(ex == "positivos"): #
                if(imagen_1[i][j] == 1 and imagen_2[i][j] == 1):
                    verdaderos_pn_1 += 1
                    pn_totales += 1
                elif(imagen_1[i][j] == 1):
                    pn_totales += 1
                if(imagen_1[i][j] == 1 and imagen_3[i][j] == 1):
                    verdaderos_pn_2 += 1
            elif(ex == "negativos"):
                if(imagen_1[i][j] == 0 and imagen_2[i][j] == 0):
                    verdaderos_pn_1 += 1
                    pn_totales += 1
                elif(imagen_1[i][j] == 0):
                    pn_totales += 1
                if(imagen_1[i][j] == 0 and imagen_3[i][j] == 0):
                    verdaderos_pn_2 += 1
            else:
                print("You have to assign an opcion between 'positivos' and 'negativos'.")
    return pn_totales, verdaderos_pn_1, verdaderos_pn_2


if __name__ == "__main__":   
    imagenes = 100
    exactitud_pos_promedio = 0
    exactitud_neg_promedio = 0
    tabla = np.zeros((imagenes,5),dtype = "float64")
    promedio_1 = []
    promedio_2 = []
    for i in range(imagenes):
        indice = i + 1

        image_1 = cv.imread("Personas/segmentacion/manual/%d.jpg"%indice,cv.IMREAD_GRAYSCALE)
        image_2 = cv.imread("Personas/segmentacion/programa/%d.jpg"%indice,cv.IMREAD_GRAYSCALE)
        image_3 = cv.imread("Personas/segmentacion/programa_2/%d.jpg"%indice,cv.IMREAD_GRAYSCALE)
       
        image_1 = image_1[30:250,85:435]
        image_2 = image_2[40:247,55:388]
        image_3 = image_3[40:247,55:388]
        
        image_1 = tr.resize(image_1,(220,360),mode='constant')
        image_2 = tr.resize(image_2,(220,360),mode = 'constant')#tr.resize() Aplica un escalamiento a la imagen 
        image_3 = tr.resize(image_3,(220,360),mode = 'constant')
        
        umbral1 = otsu(image_1)
        umbral2 = otsu(image_2)
        umbral3 = otsu(image_3)        
        """
        plt.title("Segmentada Manual {indice}".format(indice=indice))
        plt.imshow(image_1,cmap="gray")
        plt.show()
        
        plt.title("Segmentada Programa {indice}".format(indice=indice))
        plt.imshow(image_2,cmap="gray")
        plt.show()
        
        plt.title("Segmentada Programa_2 {indice}".format(indice=indice))
        plt.imshow(image_3,cmap="gray")
        plt.show()
        
        image_1 = np.array(image_1, dtype="bool") # Imagen segmentada a mano para autos
        image_2 = np.array(image_2, dtype="bool") # Imagen segmentada con el método propuesto
        image_3 = np.array(image_3, dtype="bool") # Imagen segmentada con el método del articulo
        image_4 = np.array(image_4, dtype="bool") # Imagen segmentada a mano para personas
        image_5 = np.array(image_5, dtype="bool") # Imagen segmentada con el método propuesto
        image_6 = np.array(image_6, dtype="bool") # Imagen segmentada con el método del artículo
        """
        image_1 = image_1 > umbral1 # Imagen segmentada a mano para autos
        image_2 = image_2 > umbral2 # Imagen segmentada con el método propuesto
        image_3 = image_3 > umbral3 # Imagen segmentada con el método del articulo
       
        #Métrica f1 para personas
        positivos_totales_autos,verdaderos_positivos_1,verdaderos_positivos_2 = verdaderos_PN(image_1,image_2,image_3,ex = "positivos")
        negativos_totales_autos,verdaderos_negativos_1,verdaderos_negativos_2 = verdaderos_PN(image_1,image_2,image_3,ex = "negativos") 
        
        falsos_positivos_1,falsos_positivos_2 = falsos_PN(image_1,image_2,image_3,f = "positivos")
        falsos_negativos_1,falsos_negativos_2 = falsos_PN(image_1,image_2,image_3,f = "negativos")
        
        
        
        #exactitud_pos_promedio += verdaderos_positivos_1
        #exactitud_neg_promedio += verdaderos_negativos_1
        
        #n = verdaderos_positivos_1 + verdaderos_negativos_1 + falsos_positivos_1 + falsos_negativos_1
        
        tabla[i,0] = indice
        tabla[i,1] = verdaderos_positivos_1 / (verdaderos_positivos_1 + falsos_negativos_1) #Recall
        tabla[i,2] = verdaderos_positivos_1 / (verdaderos_positivos_1 + falsos_positivos_1) #Precision
        tabla[i,3] = verdaderos_positivos_2 / (verdaderos_positivos_2 + falsos_negativos_2) 
        tabla[i,4] = verdaderos_positivos_2 / (verdaderos_positivos_2 + falsos_positivos_2)
        #Comienza cálculo de la métrica F1 a partir de los valores de la tabla
        f1_1 = 2 * ((tabla[i,1] * tabla[i,2])/(tabla[i,1] + tabla[i,2])) #F1 personas metodo propuesto
        f1_2 = 2 * ((tabla[i,3] * tabla[i,4])/(tabla[i,3] + tabla[i,4])) #F1 personas metodo articulo
        
        
        promedio_1.append(f1_1) #Promedio metodo propuesto autos
        promedio_2.append(f1_2) #Promedio metodo artículo autos
        
        
        """
        print("exactitud 1: {expp}/{extp} \n"
              "exactitud 0: {expn}/{extn} \n"
              "falsos_positivos_1: {fp} \n"
              "falsos_negativos_1: {fn} \n"
              "Métrica F1_1: {f1}".format(expp = verdaderos_positivos_1,
              extp = positivos_totales_autos,expn = verdaderos_negativos_1,extn = negativos_totales_autos,
              fp = falsos_positivos_1, fn = falsos_negativos_1,f1=f1_1))
        
       
        print("exactitud 1: {expp}/{extp} \n"
              "exactitud 0: {expn}/{extn} \n"
              "falsos_positivos_2: {fp} \n"
              "falsos_negativos_2: {fn} \n"
              "Métrica F1_2: {f1}".format(expp = verdaderos_positivos_2,
              extp = positivos_totales_autos,expn = verdaderos_negativos_2,extn = negativos_totales_autos,
              fp = falsos_positivos_2, fn = falsos_negativos_2,f1=f1_2))
        
        print("exactitud 1: {expp}/{extp} \n"
              "exactitud 0: {expn}/{extn} \n"
              "falsos_positivos_2: {fp} \n"
              "falsos_negativos_2: {fn} \n"
              "Métrica F1_2: {f1}".format(expp = verdaderos_positivos_3,
              extp = positivos_totales_personas,expn = verdaderos_negativos_3,extn = negativos_totales_personas,
              fp = falsos_positivos_3, fn = falsos_negativos_3,f1=f1_3))
       
        print("exactitud 1: {expp}/{extp} \n"
              "exactitud 0: {expn}/{extn} \n"
              "falsos_positivos_2: {fp} \n"
              "falsos_negativos_2: {fn} \n"
              "Métrica F1_2: {f1}".format(expp = verdaderos_positivos_4,
              extp = positivos_totales_personas,expn = verdaderos_negativos_4,extn = negativos_totales_personas,
              fp = falsos_positivos_4, fn = falsos_negativos_4,f1=f1_4))
       """
    promedio_1 = np.sum(promedio_1) / imagenes
    promedio_2 = np.sum(promedio_2) / imagenes