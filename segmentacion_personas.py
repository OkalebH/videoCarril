# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 23:31:23 2018

@author: Okale
"""
from math import ceil
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
    n = 4 * (np.sum(desvEstandar) / (num_pixeles - pixeles_0(desvEstandar)))
    desvEstandar += n
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
def imagen_auto_segmentada(prom_masDesvEstandar,prom_menosDesvEstandar,pintar=True):
    imagen_auto1 = np.zeros((230,350),dtype="float64")
    imagen_auto2 = np.zeros((230,350),dtype="float64")
    nombre = 0#randint(1,339)
    detect = []
    for indice in range(100):
        nombre = indice + 1 
        imagen = cv.imread("Personas/segmentacion/imagenes/%d.jpg"%nombre,cv.IMREAD_GRAYSCALE)    
        
        imagen_auto1 = imagen < prom_menosDesvEstandar    
        imagen_auto2 = imagen > prom_masDesvEstandar   
        imagen_or = np.bitwise_or(imagen_auto1,imagen_auto2)   #func_or(imagen_auto1,imagen_auto2)
        
        s = 10
        w = 3
        t = (((w-1)/2)-0.5)/s
        imagen_filtrada = gaussian(imagen_or,sigma = s,truncate = t)
        imagen_filtrada = median(imagen_or,np.ones((5,5)))
        umbral = otsu(imagen_filtrada)
        imagen_filtrada = imagen_filtrada > umbral
        
        imagen_resultante = imagen * imagen_filtrada
        num_pixeles = np.sum(imagen_filtrada)
        num_personas = num_pixeles / 1200
        num_personas_t = round(num_personas)
        detect.append(num_personas_t)
        plt.title('Imagen filtrada')
        plt.imshow(imagen_filtrada,cmap='gray')
        plt.savefig("Personas/segmentacion/programa/%d.jpg"%nombre)
        plt.show()
        print("{cont} Num pixeles = {pixeles} Num personas = {personas} round={trun}".format(cont=nombre,pixeles = num_pixeles,personas=num_personas,trun=num_personas_t))
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
            plt.title('Imagen leida %d'%nombre)
            plt.imshow(imagen,cmap='gray')
            plt.show()
            
            plt.title('Imagen filtrada')
            plt.imshow(imagen_filtrada,cmap='gray')
            plt.savefig("Personas/segmentacion/programa/%d.jpg"%nombre)
            plt.show()
            
            plt.title("Imagen segmentada")
            plt.imshow(imagen_resultante,cmap="gray")
            plt.savefig("imSegmentada_personas.jpg")
            plt.show()
            
            plt.title('Imagen leida < prom_menosDesvEstandar')
            plt.imshow(imagen_auto1,cmap='gray')
            plt.savefig("imMenor_personas.jpg")
            plt.show()
                
            plt.title('Imagen Auto > prom_masDesvEstandar')
            plt.imshow(imagen_auto2,cmap='gray')
            plt.savefig("imMayor_personas.jpg")
            plt.show()
            """
            plt.title("Imagen_or")
            plt.imshow(imagen_or,cmap="gray")
            plt.savefig("imSegmentada_personas.jpg")
            #plt.savefig("Personas/resultadoPersonas/%d.jpg"%nombre)
            plt.show() 
            
           
            
            plt.title('Histograma')
            plt.plot(histograma)
            plt.plot(linea)
            plt.text(int(umbral_rango),500,u'umbral_otsu',fontsize = 10,horizontalalignment = 'center')
            plt.axis([0,255,0,600])
            plt.show()
            """
    
    
    return imagen_resultante,detect

def leer_imagenes(filas,columnas,num_imagenes,ruta):
    image_box = np.zeros((filas,columnas,num_imagenes),dtype="float64")
    for i in range(num_imagenes):
        indice = i + 1
        image_box[:,:,i] = cv.imread(ruta%indice,cv.IMREAD_GRAYSCALE)
    
    return image_box

if __name__ == '__main__':
    filas = 230
    columnas = 350
    num_imagenes = 616    
    ruta = "Personas/no_personas/%d.jpg"
    
    image_box = leer_imagenes(filas,columnas,num_imagenes,ruta)    
    prom_masDesvEstandar, prom_menosDesvEstandar = prom_desvEstandar(image_box,pintar=False)
    imagen_p,detect = imagen_auto_segmentada(prom_masDesvEstandar,prom_menosDesvEstandar,pintar=False)
    """
    plt.title('Imagen Segmentada')
    plt.imshow(imagen_p,cmap='gray')
    plt.savefig("imSegmentada.jpg")
    plt.show()
    """