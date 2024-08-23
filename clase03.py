#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:16:55 2024

@author: Estudiante
"""
#%%
import numpy as np
import pandas as pd 

#%%

nombre_archivo = 'arbolado-en-espacios-verdes.csv'
df = pd.read_csv(nombre_archivo, index_col = 2)

#%%
Jacarandas = df[df['nombre_com'] == 'Jacarandá']
jac_cant_arboles = df.shape[0]
jac_altura_max = df['altura_tot'].max()
jac_altura_min = df['altura_tot'].min()
jac_altura_promedio = (df['altura_tot'].sum()) / cant_arboles

jac_diametro_max = df['diametro'].max()
jac_diametro_min = df['diametro'].max()
jac_diametro_promedio = (df['diametro'].sum()) / cant_arboles

#%%
Palos = df[df['nombre_com'] == 'Palo borracho rosado']
palos_cant_arboles = Palos.shape[0]
palos_altura_max = Palos['altura_tot'].max()
palos_altura_min = Palos['altura_tot'].min()
palos_altura_promedio = Palos['altura_tot'].mean()

palos_diametro_max = Palos['diametro'].max()
palos_diametro_min = Palos['diametro'].max()
palos_diametro_promedio = Palos['diametro'].mean()

#%%
def cantidad_arboles(parque):alturas(lista_arboles, especie) que,
dada una lista como la generada con leer_parque(...) y una especie de
árbol (un valor de la columna 'no
    return df[df['espacio_ve'] == parque].shape[0]

def cantidad_nativos(parque):
   return df[(df['espacio_ve'] == parque) & df(['origen'] =='Nativo/Autóctono')].shape[0]
   
#%%
# EJERCICIO 1:
def leer_parque(nombre_archivo, parque):
    df = pd.read_csv(nombre_archivo, index_col = 2)
    df[df['espacio_ve'] == parque]
    return df.to_dict()
        
#%%
# EJERCICIO 2:
def especies(lista_arboles):
    


