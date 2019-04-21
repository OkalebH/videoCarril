import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from math import ceil,floor
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
    cap = cv.VideoCapture("Videos/Personas.mp4")
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if(not ret):
            break
        gray = cv.cvtColor(frame,cv.COLOR_RGBA2GRAY)
        
        gray = gray[200:430,500:850]
        imagen_auto1 = gray < prom_menosDesvEstandar    
        imagen_auto2 = gray > prom_masDesvEstandar   
        imagen_or = np.bitwise_or(imagen_auto1,imagen_auto2)   #func_or(imagen_auto1,imagen_auto2)
        s = 10
        w = 3
        t = (((w-1)/2)-0.5)/s
        imagen_filtrada = gaussian(imagen_or,sigma = s,truncate = t)
        umbral = otsu(imagen_filtrada)
        imagen_filtrada = imagen_filtrada > umbral
        imagen_filtrada = median(imagen_or,np.ones((5,5)))
        
        
        num_pixeles = np.sum(imagen_filtrada) / 255
        num_personas = num_pixeles / 1200
        num_personas_t = round(num_personas)
        #num_ceil = ceil(num_personas)
        #num_floor = floor(num_personas)
        print("Num pixeles = {pixeles} Num personas = {personas} ceil={trun}".format(pixeles = num_pixeles,personas=num_personas,trun=num_personas_t))
        imagen_resultante = gray * imagen_filtrada
        cv.imshow("Imagen segmentada binaria",imagen_filtrada)
        cv.imshow("Video segmentado",imagen_resultante)
        cv.imshow("Video leido",frame)
        #time.sleep(.20)
        if (cv.waitKey(0) & 0xff == ord("q")):
            break
        #time.sleep(.20)
    cap.release()
    cv.destroyAllWindows()

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
    imagen_p = imagen_auto_segmentada(prom_masDesvEstandar,prom_menosDesvEstandar,pintar=True)
    """
    plt.title('Imagen Segmentada')
    plt.imshow(imagen_p,cmap='gray')
    plt.savefig("imSegmentada.jpg")
    plt.show()
    """