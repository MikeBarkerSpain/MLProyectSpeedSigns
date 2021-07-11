'''
contains the generic functionality related to open,create, read and write files
'''
import pandas as pd
import numpy as np
import lxml
import xlrd
import os, sys
from os import listdir
from os.path import isfile, join
import cv2 as cv
from PIL import Image
from PIL.ExifTags import TAGS

dir = os.path.dirname

# se saltan tres veces de directorio para llegar al directorio raíz del proyecto '/Entregable_EDA'
# la primera para eliminar el nombre del archivo 
# las dos siguientes para eliminar los directorios '/utils' y '/src'
src_path = dir(dir(dir(os.path.abspath(__file__))))

# se incorpora la ruta hasta el directorio raiz al path del archivo
sys.path.append(src_path)

def get_img_array_from_dir(data_path,img_height, img_width):
    # Generación del conjunto de X_test
    df_dict = []

    # se obtienen las imágenes de la carpeta objeto de análisis
    only_image_names = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    # cargar imágenes con 32 bits
    for image_name in only_image_names:
        if ".png" in image_name:           # se reduce el set de set (este notebook solo sirve para comprobar que el modelo se carga)   
            image_fullpath = data_path + os.sep + image_name
            image_cv = cv.imread(image_fullpath)     # con el 0 se lee en blanco y negro, sin nada, se lee en color

            # las imagenes originales tienen tamañanos distintos  -> con esta función se ponen en el tamaño común para todas (y será el tamaño de entrada al modelo)
            image_cv = cv.resize(image_cv, (img_height, img_width)) 
            df_dict.append({'image':image_cv})

    df = pd.DataFrame(df_dict)
    X_test_32 = np.stack(np.array(df["image"]))
    X_test_32_1 = X_test_32     #/255
    return X_test_32_1, only_image_names

def save_df_to_csv(final_df):
    final_df.to_csv(src_path + '/data/final_img_analisis_df.csv')

def save_df_compmodels_to_csv(final_df):
    final_df.to_csv(src_path + '/data/final_models_comp_df.csv')

def save_df_timetable_to_csv(final_df):
    final_df.to_csv(src_path + '/data/final_timetable.csv')

def read_tiempos ():
    filename = src_path + '/documentation/Timetable.csv'
    tiempos = pd.read_csv(filename, sep=';')
    return tiempos