'''
This file contains the generic functionality related tocollect data, clean data and others (wrangling methods such usworking with multiples jsons)
'''

import os, sys

dir = os.path.dirname

# se saltan tres veces de directorio para llegar al directorio raíz del proyecto '/Entregable_EDA'
# la primera para eliminar el nombre del archivo 
# las dos siguientes para eliminar los directorios '/utils' y '/src'
src_path = dir(dir(os.path.abspath(__file__)))

# se incorpora la ruta hasta el directorio raiz al path del archivo
sys.path.append(src_path)

import pandas as pd
import numpy as np

import utils.folders_tb as fold

def voting3_pred (img_list,img_height, img_width, model0, model11, model12, model21, model22):
    ### PREDICCIONES ####
    # Variables de predicción de señal (tanto final como cada modelo)
    Y_pred = []         # predicción final 
    Y_pred_mod0 = []    # predicción modelo 0
    Y_pred_mod1 = []    # predicción modelo 1
    Y_pred_mod2 = []    # predicción modelo 2


    for i in range(len(img_list)):
        # imagen a analizar
        img_analisis = img_list[i].reshape(1,img_height, img_width,3)

        ### PREDICCIONES MODELO 0 ###
        pred_mod0 = np.argmax(model0.predict(img_analisis))
        Y_pred_mod0.append(pred_mod0)

        ### PREDICCIONES MODELO 1 ###
        predictions_vgg16_class1 = model11.predict(img_analisis)
        if np.argmax(predictions_vgg16_class1) == 2:
            predictions_vgg16_signs1 = model12.predict(img_analisis)

            # se añade 1 a la clase de la señal ya que este submodelo no tiene en cuenta la clase 'no_speed'
            pred_mod1 = np.argmax(predictions_vgg16_signs1)+1
            Y_pred_mod1.append(pred_mod1)      
        else:
            Y_pred_mod1.append(0)
        
        ### PREDICCIONES MODELO 2 ###
        predictions_vgg16_class2 = model21.predict(img_analisis)
        if np.argmax(predictions_vgg16_class2) == 2:
            predictions_vgg16_signs2 = model22.predict(img_analisis)
            pred_mod2 = np.argmax(predictions_vgg16_signs2)
            Y_pred_mod2.append(pred_mod2)
        else:
            Y_pred_mod2.append(0)

        ### PREDICCIÓN CONJUNTA ######
        Y_pred.append([Y_pred_mod0[i], Y_pred_mod1[i], Y_pred_mod2[i]])
    return Y_pred,Y_pred_mod0,Y_pred_mod1,Y_pred_mod2

        