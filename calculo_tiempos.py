# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 22:17:01 2018

@author: Okale
"""
from time import sleep
from random import randint as rnd
from math import ceil
C_p = 8
C = 10
xPersonas = 1.4
xAutos = 10
yAutos = 10
tiempoPersonas = 0
tiempoAutos = 0

for i in range(15):
    NP = i + 1#rnd(0,15)
    NA = i + 1#rnd(0,10)
    print('Personas: {}'.format(NP))
    print('Autos: {}'.format(NA))
    if(NP >= 6):
        tiempoPersonas = xPersonas * NP
    else:
        if(NP != 0):
            tiempoPersonas = C_p
        else:
            tiempoPersonas = 0
        
    if(NA >= 2):
        tiempoAutos = (C * NA / 3) + yAutos
    else:
        if(NA != 0):
            tiempoAutos = C
        else:
            tiempoAutos = 0
    tiempoPersonas = ceil(tiempoPersonas)
    tiempoAutos = ceil(tiempoAutos)
    
    print("El tiempo de las personas es: {}".format(tiempoPersonas))
    print("El tiempo de las autos es: {} \n".format(tiempoAutos))
    #sleep(5)
    