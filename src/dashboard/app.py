import streamlit as st
import os,sys

    
path = os.path.dirname(os.path.abspath(__file__))
dir = os.path.dirname
src_path = dir(dir(os.path.abspath(__file__)))


sys.path.append(src_path)

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # se sube hasta el nivel de la raiz del proyecto

import utils.dashboard_tb as dsh


dsh.configuracion()

# cargar datos del análisis de las imágenes
csv_path = root_path + os.sep + 'data' + os.sep + 'final_img_analisis_df.csv'
df = dsh.cargar_datos(csv_path)

# cargar datos del análisis de los modelos
csv_path_model = root_path + os.sep + 'data' + os.sep + 'final_models_comp_df.csv'
df_model = dsh.cargar_datos(csv_path_model)

menu = st.sidebar.selectbox('Menu:',
                            options=['Bienvenida','Proyecto','Predicciones', 'Predicción Real', 'Imágenes y Predicciones','Comparativa de modelos','Tabla de tiempos'])

st.title("Proyecto de Machine Learning Reconocimiento de señales de velocidad en autopistas")

if menu == 'Bienvenida':    # Explicación del dataset referencias e hipótesis
    dsh.menu_home()
elif menu == 'Proyecto':   # Explicación del planteamiento del análisis y voting
    dsh.menu_visu()
elif menu == 'Predicciones':    # Predicciones de cada modelo e introducir la solución de voting
    dsh.menu_predict()
elif menu == 'Predicción Real':    # Predicciones de cada modelo, e incluir una predicción en caliente del modelo
    dsh.menu_predict_real()
elif menu == 'Imágenes y Predicciones':    # salida del df con los datos de análisis de las imágenes
    dsh.menu_flask(df)
elif menu == 'Comparativa de modelos':    # Tabla de comparaciones de los modelos
    dsh.menu_modelo_comp(df_model)
elif menu == 'Tabla de tiempos':    # mostrar los tiempos de cada concepto del proyecto
    dsh.menu_timetable()