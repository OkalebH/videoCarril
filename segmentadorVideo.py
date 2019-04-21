# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 00:34:51 2019

@author: Okale
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#import random as rnd
from skimage.filters import gaussian
from skimage.measure import label
from skimage.filters import threshold_otsu as otsu
from skimage.filters.rank import median

def prom_desvEstandar(image_box,pintar=True):
    #Definición de arreglos que contendrán las imágenes resultantes
    promedio = np.zeros((image_box.shape[0],image_box.shape[1]),dtype="float64")
    desvEstandar = np.zeros((image_box.shape[0],image_box.shape[1]),dtype="float64")
    prom_masDesvEstandar = np.zeros((image_box.shape[0],image_box.shape[1]),dtype="float64")
    prom_menosDesvEstandar = np.zeros((image_box.shape[0],image_box.shape[1]),dtype="float64")
    num_imagenes = image_box.shape[2]
    num_pixeles = image_box.shape[0] * image_box.shape[1]
    """
    Obtiene las imagenes y hace la sumatoria para calcular el promedio
    """
    promedio = np.sum(image_box,axis = 2) / num_imagenes
    """
    Calcula la desviación estándar
    """
    for i in range(num_imagenes):
        desvEstandar += (image_box[:,:,i] - promedio) ** 2
    
    desvEstandar = np.sqrt(desvEstandar / (num_imagenes - 1))
    """Es una forma de calcular la variable n tomando en cuenta factores de la imagen"""
    n = 4 * (np.sum(desvEstandar) / (num_pixeles - pixeles_0(desvEstandar)))
    desvEstandar += n
    """Aproxima los pixeles de las imágenes a los puntos de inflexión de la campana Gaussiana"""
    prom_masDesvEstandar = promedio + desvEstandar   
    prom_menosDesvEstandar = promedio - desvEstandar    
    if(pintar == True):
        plt.title('Promedio')
        plt.imshow(promedio,cmap='gray')
        plt.savefig("imProm.jpg")
        plt.show()
        
        plt.title('Desviación Estándar')
        plt.imshow(desvEstandar,cmap='gray')
        plt.savefig("imDesv.jpg")
        plt.show()
        
        plt.title('Promedio + Desviación Estándar')
        plt.imshow(prom_masDesvEstandar,cmap='gray')
        plt.savefig("imMas.jpg")
        plt.show()
        
        plt.title('Promedio - Desviación Estándar')
        plt.imshow(prom_menosDesvEstandar,cmap='gray')
        plt.savefig("imMenos.jpg")
        plt.show()
        
    return prom_masDesvEstandar,prom_menosDesvEstandar
def pixeles_0(imagen):
    cont = 0
    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
            if(imagen[i][j] == 0):
                cont += 1
    return cont
"""Funcion que abre el video y realiza el segmentado"""
def imagen_auto_segmentada(prom_masDesvEstandar,prom_menosDesvEstandar,pintar=True):
    imagen_auto1 = np.zeros((220,360),dtype="float64")
    imagen_auto2 = np.zeros((220,360),dtype="float64")
    
    cap = cv.VideoCapture("Videos/Autos.mp4")
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if(not ret):
            break
        gray = cv.cvtColor(frame,cv.COLOR_RGBA2GRAY)
        
        gray = gray[100:320,10:370]
        imagen_auto1 = gray < prom_menosDesvEstandar    
        imagen_auto2 = gray > prom_masDesvEstandar   
        imagen_or = np.bitwise_or(imagen_auto1,imagen_auto2)   #func_or(imagen_auto1,imagen_auto2)
      
        imagen_filtrada = median(imagen_or,np.ones((10,10)))
        
        region1 = imagen_filtrada[0:60,0:360]
        region2 = imagen_filtrada[61:120,0:360]
        region3 = imagen_filtrada[121:220,0:360]
        pix_1 = np.sum(region1) / 255
        pix_2 = np.sum(region2) / 255
        pix_3 = np.sum(region3) / 255
        #print(num_pixeles)
        num_1 = pix_1 / 1600
        num_2 = pix_2 / 3000
        num_3 = pix_3 / 5400
        num_autos = num_1 + num_2 + num_3
        redondeo = round(num_autos)
        print("pix_1 = {pixeles} Num autos = {autos}".format(pixeles = pix_1,autos=num_1))
        print("pix_2 = {pixeles} Num autos = {autos}".format(pixeles = pix_2,autos=num_2))
        print("pix_3 = {pixeles} Num autos = {autos}".format(pixeles = pix_3,autos=num_3))
        print("Num_autos = {autos} redondeo = {red}".format(autos=num_autos,red=redondeo))
        imagen_resultante = gray * imagen_filtrada
        cv.imshow("Imagen segmentada binaria",region1)
        cv.imshow("Imagen segmentada",region2)
        cv.imshow("Video segmentado",region3)
        cv.imshow("Video leido",frame)
        #time.sleep(.20)
        if (cv.waitKey(0) & 0xff == ord("q")):
            break
        #time.sleep(.20)
    cap.release()
    cv.destroyAllWindows()
    

"""Lee las imágenes en la ruta especificada y las guarda en un arreglo de imágenes"""
def leer_imagenes(filas,columnas,num_imagenes,ruta):
    image_box = np.zeros((filas,columnas,num_imagenes),dtype="float64")
    for i in range(num_imagenes):
        indice = i + 1
        image_box[:,:,i] = cv.imread(ruta%indice,cv.IMREAD_GRAYSCALE)
    
    return image_box

if __name__ == '__main__':
    filas = 220
    columnas = 360
    num_imagenes = 63
    ruta = "Autos/no_autos/%d.jpg"
    image_box = leer_imagenes(filas,columnas,num_imagenes,ruta)    
    prom_masDesvEstandar, prom_menosDesvEstandar = prom_desvEstandar(image_box,pintar=False)
    imagen_auto_segmentada(prom_masDesvEstandar,prom_menosDesvEstandar,pintar=False)