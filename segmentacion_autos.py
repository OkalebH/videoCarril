# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 21:37:03 2018

@author: Okaleb
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#import random as rnd
from skimage.filters import gaussian
#from skimage.measure import label
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
    """
    histogramaDesv = np.zeros((256,1),dtype = np.double)
    filas,columnas = desvEstandar.shape
    for i in range(filas):
        for j in range(columnas):
            histogramaDesv[int(desvEstandar[i,j])][0] += 1
    plt.plot(histogramaDesv)
    plt.axis([0,50,0,40000])
    plt.show()
    """
    if(pintar == True):
        plt.title('Promedio')
        plt.imshow(promedio,cmap='gray')
        plt.savefig("imProm_autos.jpg")
        plt.show()
        
        plt.title('Desviación Estándar')
        plt.imshow(desvEstandar,cmap='gray')
        plt.savefig("imDesv_autos.jpg")
        plt.show()
        
        plt.title('Promedio + Desviación Estándar')
        plt.imshow(prom_masDesvEstandar,cmap='gray')
        plt.savefig("imMas_autos.jpg")
        plt.show()
        
        plt.title('Promedio - Desviación Estándar')
        plt.imshow(prom_menosDesvEstandar,cmap='gray')
        plt.savefig("imMenos_autos.jpg")
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
    nombre = 0#randint(1,339)
    detect = []
    for indice in range(100):
        nombre = indice + 1 
        imagen = cv.imread("Autos/segmentacion/imagenes/%d.jpg"%nombre,cv.IMREAD_GRAYSCALE)    
        
        imagen_auto1 = imagen < prom_menosDesvEstandar    
        imagen_auto2 = imagen > prom_masDesvEstandar   
        imagen_or = np.bitwise_or(imagen_auto1,imagen_auto2)   #func_or(imagen_auto1,imagen_auto2)
        
        s = 10
        w = 5
        t = (((w-1)/2)-0.5)/s
        imagen_filtrada = gaussian(imagen_or,sigma = s,truncate = t)
        
        imagen_filtrada = median(imagen_or,np.ones((10,10)))
        umbral = otsu(imagen_filtrada)      
        
        plt.title("Imagen_or_umbralada")
        plt.imshow(imagen_filtrada,cmap="gray")
        plt.savefig("Autos/segmentacion/programa/%d.jpg"%nombre)
        plt.show() 
        imagen_filtrada = imagen_filtrada > umbral
        #print(imagen_filtrada.shape)
        region1 = imagen_filtrada[0:60,0:360]
        region2 = imagen_filtrada[61:120,0:360]
        region3 = imagen_filtrada[121:220,0:360]
        pix_1 = np.sum(region1) 
        pix_2 = np.sum(region2)
        pix_3 = np.sum(region3)
        #print(num_pixeles)
        num_1 = pix_1 / 1600
        num_2 = pix_2 / 3000
        num_3 = pix_3 / 5400
        num_autos = num_1 + num_2 + num_3
        redondeo = round(num_autos)
        detect.append(redondeo)
        print("pix_1 = {pixeles} Num autos = {autos}".format(pixeles = pix_1,autos=num_1))
        print("pix_2 = {pixeles} Num autos = {autos}".format(pixeles = pix_2,autos=num_2))
        print("pix_3 = {pixeles} Num autos = {autos}".format(pixeles = pix_3,autos=num_3))
        print("Num_autos = {autos} redondeo = {red}".format(autos=num_autos,red=redondeo))
        
        
        """
        umbral_rango = umbral * 255
        rango_imagen_filtrada = imagen_filtrada * 255
        rango_imagen_filtrada = np.array(rango_imagen_filtrada,dtype=np.uint8)
        
        histograma = np.zeros((256,1),dtype = np.double)
        filas,columnas = rango_imagen_filtrada.shape
        for i in range(filas):
            for j in range(columnas):
                histograma[rango_imagen_filtrada[i,j]][0] += 1
        
        linea = np.zeros((256,1),dtype = np.double)
        linea[int(umbral_rango)] = 600
        """
        if(pintar == True):
            plt.title('Imagen Auto leida %d'%nombre)
            plt.imshow(imagen,cmap='gray')
            plt.show()
            
            plt.title('Imagen Auto < prom_menosDesvEstandar')
            plt.imshow(imagen_auto1,cmap='gray')
            plt.savefig("imMenor_autos.jpg")
            plt.show()
                
            plt.title('Imagen Auto > prom_masDesvEstandar')
            plt.imshow(imagen_auto2,cmap='gray')
            plt.savefig("imMayor_autos.jpg")
            plt.show()
            
            plt.title("Imagen_or_umbralada")
            plt.imshow(imagen_filtrada,cmap="gray")
            plt.savefig("Autos/segmentacion/programa/%d.jpg"%nombre)
            plt.show() 
            imagen_filtrada = imagen_filtrada > umbral
            num_pixeles = np.sum(imagen_filtrada)
            num_autos = num_pixeles / 4300
            redondeo = round(num_autos)
            print("Num pixeles = {pixeles} Num autos = {autos} redondeo={red}".format(pixeles = num_pixeles,autos=num_autos, red=redondeo))
        
            plt.title("Imagen_or")
            plt.imshow(imagen_or,cmap="gray")
            plt.savefig("imOr_autos.jpg")
            plt.show() 
            """
            plt.title('Histograma')
            plt.plot(histograma)
            plt.plot(linea)
            plt.text(int(umbral_rango),500,u'umbral_otsu',fontsize = 10,horizontalalignment = 'center')
            plt.axis([0,255,0,600])
            plt.show()
            """
    imagen_resultante = imagen * imagen_filtrada
    
    return imagen_resultante, detect

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
    imagen_auto,detect = imagen_auto_segmentada(prom_masDesvEstandar,prom_menosDesvEstandar,pintar=False)
    """
    plt.title('Imagen Auto Segmentada')
    plt.imshow(imagen_auto,cmap='gray')
    plt.savefig("imSegmentada_autos.jpg")
    plt.show()
    """