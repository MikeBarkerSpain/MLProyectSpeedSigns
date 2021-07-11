'''
This file contains the functionality related to the dashboard
'''

import streamlit as st
import requests as rq
from PIL import Image
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns 
import numpy as np 
import os
import random
import statistics

root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))   # se sube hasta el nivel de la raiz del proyecto

def configuracion():
    st.set_page_config(page_title='Proyecto de Machine Learning Reconocimiento de señales de velocidad en autopistas', page_icon=':votes:', layout="wide")

# 'Bienvenida': Explicación del dataset referencias e hipótesis
def menu_home():

    with st.beta_expander("Contexto"):
        st.write(""" El peatón 99'99 quiere comprar un coche que pueda mantener la velocidad en autopista si activa el control de velocidad,
         pero no está muy seguro de que los distintos elementos de los coches actuales puedan detectar correctamente las distintas señales de la
         limitación de velocidad, por lo que acude a un amigo Data Scientist para que le explique como funcionan los distintos procesos y comprobar 
         (o no) que los sistemas de control de velocidad automatizados están a la altura de sus expectativas.""")

    with st.beta_expander("Hipótesis inicial"):
        st.write("""Dado que el peatón 99'99 tiene un amigo DS, tras una breve llamada para hablar con su amigo acerca de los modelos de predicción 
        y deep learning, decide que lo mejor es ir al laboratorio para hacer algunas pruebas y ver cómo funcionan esos modelos de predicción.        
        Al llegar al laboratorio de su amigo DS, el peatón 99'99 lo tiene muy claro, debe investigar como se combinan esos modelos y comprobar
        si funcionan mejor unidos o por separado. 
        
        La hipótesis inicial del peatón 99'99 es que los modelos funcionan mejor combinados que por separado.""")
    
    # Explicación del dataset
    st.header('Explicación del dataset')
    st.write('''Lo primero que hay que tener en cuenta en este proyecto es el origen de las imágenes que se han utilizado para entrenar a los modelos,
    Estas imágenes provienen de un dataset llamado GTSRB, y  proviene de un estudio de la IEEE para las señales de tráfico Europeas.
    
    Se pueden obtener más referencias en:
    https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
    https://ieeexplore.ieee.org/document/8558481''')

    # Introducción al meta del dataset
    st.subheader('Meta del dataset')
    st.write('''Este dataset tiene un meta de 42 imágenes, lo que quiere decir que se hay imágenes de 42 señales distintas y se han tomado con distintas
    iluminaciones, ángulos, enfoques, con el fin de hacer entrenamientos robustos de los modelos.
    Estas son las imágenes que componen el dataset, que como se puede observar, son perfectamente válidas para reforzar o refutar la hipótesis inicial.''')

    # imagen de las señales de tráfico incluidas en el meta del data set
    img = Image.open(root_path + os.sep + 'resources' + os.sep + 'MetaGTSRB.png')
    st.image(img, use_column_width='auto')

    # explicación de la clasificación de señales
    st.subheader('Clasificación de las señales de tráfico en Europa')
    st.write('''Las señales de tráfico están agrupadas por categorías, y cada categoría tiene asociados un concepto, con una forma y colores distinctivos,
    de forma que son fácilmente reconocibles visualmente y los conductores pueden ajustar su conducción rápidamente.''')

    # imagen de la clasificación de las señales de tráfico 
    img1 = Image.open(root_path + os.sep + 'resources' + os.sep + 'signs_classification.png')
    st.image(img1, use_column_width='auto')

    # ejemplos de señales
    st.write('''Ejemplos de señales de tráfico en entorno real''')

    # imagen de la clasificación de las señales de tráfico 
    img2 = Image.open(root_path + os.sep + 'resources' + os.sep + 'signs_classification_2.png')
    st.image(img2, use_column_width='auto')

    if st.button("Solución propuesta por peatón 99'99 y DS"):
        with st.empty():
            st.write("""Se deben encontrar clasificaciones intermedias de las señales, y enseñar a los modelos a clasificar las señales primero 
            y luego identificar individualmente las señales que sean de velocidad, y posteriormente, comparar los resultados con los obtenidos por 
            el modelo de identificación directa de señales.""")

@st.cache(suppress_st_warning=True)
def cargar_datos(csv_path):
    df = pd.read_csv(csv_path, sep=',')
    return df

#  'Proyecto':   Explicación del planteamiento del análisis y voting
def menu_visu():
    st.subheader('Explicación de los distintos planteamientos para las predicciones')
    st.write('''Una vez definida la hipótesis inicial y el planteamiento para refutarla o respaldarla, hay que plantear 
    los entrenamientos antes de poder hacer predicciones. Con este fin, se plantean 3 modelos distintos con sus 3 entrenamientos y etiquetas correspondientes.''')

    # Explicación del modelo 0
    st.subheader('Modelo 0: opción de identificación directa')
    st.write('''Este modelo es el básico y va a entrenarse para identificar las señales de tráfico de velocidad directamente, sin un filtro de clasificación. 
    Las etiquetas se distribuyen así:''')

    # imagen de la clasificación de señales en el modelo 0
    img = Image.open(root_path + os.sep + 'resources' + os.sep + 'signs_0.jpg')
    st.image(img, use_column_width='auto')

    # Explicación del modelo 1
    st.subheader('Modelo 1: opción de clasificación e identificación por velocidad')
    st.write('''Este modelo tiene un submodelo de clasificación que distingue las señales de velocidad de las que tiene rojo en su señal y del resto de señales 
     y posteriormente identifica la señal de velocidad con su límite de velocidad. Las etiquetas se distribuyen así:''')

    # imagen de la clasificación de señales en el modelo 1
    img = Image.open(root_path + os.sep + 'resources' + os.sep + 'signs_1.jpg')
    st.image(img, use_column_width='auto')

    # Explicación del modelo 2
    st.subheader('Modelo 2: opción de clasificación e identificación por forma (las señales de velocidad tienen un aro rojo alrededor)')
    st.write('''Este modelo tiene un submodelo de clasificación que distingue las señales con un aro rojo alrededor de las que no lo tienen y del resto de señales 
    y posteriormente identifica las señales de velocidad con su límite de velocidad. Las etiquetas se distribuyen así:''')

    # imagen de la clasificación de señales en el modelo 2
    img = Image.open(root_path + os.sep + 'resources' + os.sep + 'signs_2.jpg')
    st.image(img, use_column_width='auto')


    # StopPoint
    st.subheader('Entrenamiento de los modelos')
    st.markdown("![Alt Text](https://im7.ezgif.com/tmp/ezgif-7-d5fbb4c10f20.webp)")
    st.write('''Una vez entrenados los modelos, se pueden realizar las distintas predicciones para comprobar los resultados de los modelos''')
        
    st.error('''En este punto, hay que comprobar los resultados de los modelos por separado con las imágenes de test del dataset.''')

#  'Predicciones Laboratorio':   # Predicciones de cada modelo e introducir la solución de voting
def menu_predict():
    # Introducción de los resultados de los distintos modelos
    st.subheader('Predicciones realizadas con las imágenes de test del dataset')
    st.write('''Una vex entrenados los modelos, se deben realizar las predicciones pertinentes para analizar los resultados y 
    comprobar si corresponden con los resultados esperados. En este caso, la hipótesis inicial expone que los modelos con un submodelos de clasificación
    obtendrán mejores predicciones que el modelo entrenado para identificar directamente las señales.''')

    # Resultados con las señales de velocidad con el modelo 0
    st.subheader('Modelo 0: opción de identificación directa')
    st.write('''Tal y como se puede apreciar, el modelo 0 tiene 20 fallos en una muestra de 56 imágenes, lo que resulta en un porcentaje de fallo del 35%.''')

    # imagen de la identificación de señales en el modelo 0
    img = Image.open(root_path + os.sep + 'reports' + os.sep + 'speed_model0.png')
    st.image(img, use_column_width='auto')

    # Resultados con las señales de velocidad con el modelo 1
    st.subheader('Modelo 1: opción de clasificación e identificación por velocidad')
    st.write('''Tal y como se puede apreciar, el modelo 1 tiene 18 fallos en una muestra de 56 imágenes, lo que resulta en un porcentaje de fallo del 32,14%.''')

    # imagen de la identificación de señales en el modelo 1
    img = Image.open(root_path + os.sep + 'reports' + os.sep + 'speed_model1.png')
    st.image(img, use_column_width='auto')

    # Resultados con las señales de velocidad con el modelo 2
    st.subheader('Modelo 2: opción de clasificación e identificación por forma (las señales de velocidad tienen un aro rojo alrededor)')
    st.write('''Tal y como se puede apreciar, el modelo 2 tiene 18 fallos en una muestra de 56 imágenes, lo que resulta en un porcentaje de fallo del 32,14%.''')

    # imagen de la identificación de señales en el modelo 2
    img = Image.open(root_path + os.sep + 'reports' + os.sep + 'speed_model2.png')
    st.image(img, use_column_width='auto')
        
    st.info('''Dado que los resultados de los modelos de clasificación-identificación son superiores, la hipótesis inicial ha sido reforzada,
     los modelos predicen mejor con una fase de clasificación previa y otra de identifación.''')

    st.error('''Sin embargo, tanto el peatón 99'99 el amigo DS se preguntaban si fuese posible mejorar aún más los resultados, y tras pensarlo un poco...''')

    if st.button('Solución'):
        st.markdown("![Alt Text](https://im7.ezgif.com/tmp/ezgif-7-0786ac14f384.webp)")

        st.write('''Una vez concluido que ninguno de los modelos combinados era mejor que otro, al amigo DS se le ocurrió una solución:
        Aplicar el método de voting a los resultados obtenidos por los distintos modelos por separado. ''')

        # Explicación del modelo combinado
        st.subheader('Modelo Combinado:')
        st.subheader('Este modelo identifica las señales en función de los resultados de los modelos 0, 1 y 2')
        st.write('''El modelo combinado basa su respuesta en los resultados intermedios de los modelos anteriores, para ello, toma los resultados y 
        escoge como resultado final el más común entre los resultados intermedios, por lo tanto, es de esperar que el resultado final tenga una mayor eficiencia 
        que el de los modelos por separado.''')

        # Esquema de funcionamiento del modelo combinado con voting
        img = Image.open(root_path + os.sep + 'resources' + os.sep + 'VotingSchema.png')
        st.image(img, use_column_width='auto')

        # Resultados con las señales de velocidad con el modelo combinado
        st.subheader('Modelo Combinado: identificación de señales de velocidad en el conjunto de test.')
        st.write('''Tal y como se puede apreciar, el modelo combinado tiene 17 fallos en una muestra de 56 imágenes, lo que resulta en un porcentaje de fallo del 30,36%,
         lo que supone una mejora con respecto de cada modelo por separado.''')

        # imagen de la identificación de señales con el modelo combinado
        img = Image.open(root_path + os.sep + 'reports' + os.sep + 'speed_model.png')
        st.image(img, use_column_width='auto')

        # Uso del modelo combinado con imágenes del entorno real
        st.subheader('Uso del modelo en un entorno real')
        st.write('''Una vez comprobado que el modelo identifica las señales de velocidad, se decide usarlo con imágenes de un entorno real, con el fin de
        comprobar si esa eficiencia se mantiene o si por el contrario se modifica debido a las condiciones de conducción. ''')

        # imagen de los resultados del modelo combinado con imágenes reales
        img = Image.open(root_path + os.sep + 'reports' + os.sep + 'real_model.png')
        st.image(img, use_column_width='auto')

        st.write("""Como se puede comprobar, en las imágenes de entorno real el modelo combinado tiene una efectividad mucho peor que con las imágenes 
        del conjunto de test (en la imagen superior hay 2 aciertos contra 6 fallos, y lo que es más grave, un falso negativo -hay una señal de límite 
        de 100 km/h que se ha reconocido como ‘no_speed’).""")
        
    if st.button('Conclusiones'):
        st.subheader('''Los modelos con submodelos funcionan mejor, y si se combinan entre ellos, el resultado es mucho mejor que las predicciones por separado.''')      
        st.error('''Sin embargo, el modelo combinado no cumple con las expectativas, por lo tanto es de suponer que peatón 99'99 y amigo DS lo mejoren; pero eso es
        otra historia...''')
        st.info('MUCHAS GRACIAS POR VUESTRA ATENCIÓN')

# Predicciones Reales
def menu_predict_real ():   #  Predicciones del modelo final, e incluir una predicción en caliente del modelo
    pass

# 'Flask-Predicciones': salida del df con los datos de análisis de las imágenes
def menu_flask(df):
    st.subheader("Selección de imágenes analizadas en el proyecto")
    st.dataframe(df)

    if st.button('Get data'):
        datos_json = rq.get('http://localhost:6060/pwd?password=T05290575').json()
        st.dataframe(pd.DataFrame(datos_json))

# 'Comparativa de modelos':  Tabla de comparativas entre modelos 
def menu_modelo_comp(df_model):
    st.subheader("Comparativa de modelos con resultados de diferentes escenarios")
    st.dataframe(df_model)

# 'Tabla de tiempos':    #mostrar los tiempos de cada concepto del proyecto
def menu_timetable():
    pass
'''
def menu_filtrado(df):

    st.sidebar.subheader("Filtros:")

    distritos, filtro_distrito, barrios, filtro_barrio, censo, filtro_censo, partido_ant, filtro_partido_ant, visu = opciones_filtros(df)

    df_filtrado = filtrar(df, distritos, filtro_distrito, barrios, filtro_barrio, censo, filtro_censo, partido_ant, filtro_partido_ant)

    if df_filtrado.shape[0] == 0:
        st.warning("Los filtros que has seleccionado no nos devuelven ningun valor.")
        st.stop()

    graficas(df_filtrado, filtro_distrito, filtro_barrio, filtro_censo, filtro_partido_ant, visu)

def opciones_filtros(df):

    visu = st.sidebar.radio("¿Deseas ver los datos separados por partido?",('No', 'Si'))

    distritos = st.sidebar.multiselect(
        'Selecciona el distrito que te interese:',
        options=df.Distritos.unique().tolist())

    filtro_distrito = st.sidebar.checkbox('Quiero filtrar por distrito')
    
    barrios = st.sidebar.multiselect(
        'Selecciona los barrios que te interesen:',
        options=df.Barrios.unique().tolist())

    filtro_barrio = st.sidebar.checkbox('Quiero filtrar por barrio')

    censo = st.sidebar.select_slider("Selecciona el censo",
                                      options=range(min(df.iloc[:, 6]), max(df.iloc[:, 6]) + 1),
                                      value=(min(df.iloc[:, 6]), max(df.iloc[:, 6])))

    filtro_censo = st.sidebar.checkbox('Quiero filtrar por censo')

    partido_ant = st.sidebar.selectbox(
        'Selecciona partido:',
        options=df.anterior.unique().tolist())
    filtro_partido_ant = st.sidebar.checkbox('Quiero filtrar por el partido que ganó en las elecciones anteriores')

    return distritos, filtro_distrito, barrios, filtro_barrio, censo, filtro_censo, partido_ant, filtro_partido_ant, visu

    
    
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 12,
        }
    if visu == 'No':
        fig, ax = plt.subplots(figsize=(10,10))
        g = sns.lmplot(data=df_filtrado, x="Locales", y="D-PP", height=6, aspect=9/6)
        stat = stats.pearsonr(df_filtrado['D-PP'], df_filtrado['Locales'])
        stat_r = round(stat[0], 3)
        plt.text(200, 0, f'Correlación de Pearson: {stat_r}', fontdict=font)
        plt.show()
        st.pyplot(g)
    else:
        fig, ax = plt.subplots(figsize=(5,5))
        g = sns.lmplot(data=df_filtrado, x="Locales", y="D-PP", hue='anterior',palette="Set1", height=6, aspect=9/6)
        stat = stats.pearsonr(df_filtrado['D-PP'], df_filtrado['Locales'])
        stat_r = round(stat[0], 3)
        plt.text(200, 0, f'Correlación de Pearson: {stat_r}', fontdict=font)
        plt.show()
        st.pyplot(g)
'''