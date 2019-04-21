# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 02:38:21 2019

@author: Okale
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def leer_imagenes(filas,columnas,num_imagenes,ruta):
    image_box = np.zeros((filas,columnas,num_imagenes),dtype="float64")
    indice = 0
    i = 0
    while(i < num_imagenes ):
        indice +=2
        image_box[:,:,i] = cv.imread(ruta%indice,cv.IMREAD_GRAYSCALE)
        i += 1
    return image_box

def clasificacion_fondo(image_box,R,Min):
    indice = 0
    filas = image_box.shape[0]
    columnas = image_box.shape[1]
    ruta = "Personas/segmentacion/imagenes/%d.jpg"
    imagen_entrada = np.array((filas,columnas),dtype="float64")
    imagen_segmentada = np.zeros((filas,columnas),dtype="float64")
    clasificador = np.zeros((filas,columnas),dtype="float64")
    for i in range(1):
        indice += 1
        imagen_entrada = cv.imread(ruta%indice,cv.IMREAD_GRAYSCALE)
        plt.title('Imagen Entrada')
        plt.imshow(imagen_entrada,cmap='gray')
        plt.show()
        for imagen in range(20):            
            clasificador += np.abs(imagen_entrada - image_box[:,:,imagen]) < R
            imagen_segmentada = clasificador < Min
        clasificador = 0
        plt.title("Imagen_segmentada")
        plt.imshow(imagen_segmentada,cmap='gray')
        plt.savefig("Personas/segmentacion/programa_2/%d.jpg"%indice)
        plt.show()
    return imagen_segmentada#clasificador
    

if __name__ == "__main__":
    R = 32
    Min = 18
    filas = 230
    columnas = 350
    num_imagenes = 20    
    ruta = "Personas/no_personas/%d.jpg"
    
    image_box = leer_imagenes(filas,columnas,num_imagenes,ruta)    
    clasified_image = clasificacion_fondo(image_box,R,Min)
