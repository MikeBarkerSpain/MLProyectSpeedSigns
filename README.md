# MLProyectSpeedSigns
Proyecto de Machine Learning para el reconocimiento de señales de velocidad en autopistas

Abstract
Este proyecto pretende generar un modelo de reconocimiento de imágenes que permita analizar una imagen e identificar si contiene una señal de velocidad, y si fuese el caso, devolver la velocidad a la que se tiene que ajustar la conducción. 


Hipótesis del proyecto
La hipótesis de este proyecto es demostrar que es mejor realizar un modelo compuesto de varios submodelos en vez de utilizar un único modelo para reconocer las señales de velocidad. 
Para ello, se pretende subdividir las señales en distintas clases, y entrenar varios modelos para identificar esas clases intermedias y las señales finales. Estas son las tres opciones que se van a analizar:
1.	Opción 0: un único modelo que aprenda con tantas clases como tipos de señales distintas haya en el dataset, será el modelo a tener en cuenta como referencia.
2.	Opción 1: un modelo compuesto por dos submodelos, un primer submodelo entrenado con tres clases (las señales de velocidad, el resto de señales con rojo y el resto de señales del dataset), y un segundo submodelo entrenado para identificar de cual de las señales de velocidad se trata.
3.	Opción 2: un modelo compuesto por dos submodelos, un primer submodelo entrenado con tres clases (las señales con aro rojo, el resto de señales con rojo y el resto de señales del dataset), y un segundo submodelo entrenado para distinguir si es una señale de prohibición que no sea de velocidad y en caso de ser de velocidad, identificar de cual de las señales de velocidad se trata.


Introducción al proyecto
Para poder desarrollar este proyecto, ha sido necesario buscar los datos necesarios, es decir, un dataset de señales de tráfico lo suficientemente importante como para que una red neuronal pueda entender las imágenes y generar los patrones suficientes para identificar las señales de tráfico y sus distintos matices. 
Gracias al estudio realizado por la IEEE, las referencias a varios datasets de señales de tráfico europeas han sido aglutinadas en un mismo documento y se han clasificado las señales para que se puedan entender mejor.
Objetivo del proyecto
El objetivo del proyecto es crear una red neuronal que pueda identificar las señales de velocidad de una autopista y por lo tanto controlar la velocidad de un coche. 
Para ello, se realizarán varios modelos y se compararán sus distintos resultados:
•	Entrenar el modelo con la totalidad de imágenes en un solo paso e insertar una imagen para ver si la identifica correctamente (OPCIÓN 0).
•	Entrenar el modelo en dos tiempos con dos opciones posibles:
1.	En un primer paso entrenarle para que discierna entre las señales de velocidad, señales de peligro y otras señales; y en un segundo paso entrenarle para que identifique las distintas señales de velocidad (OPCIÓN 1).
2.	En un primer paso enseñarle para que discierna entre las señales de velocidad y prohibición (circulares con marco rojo), señales de alerta (señales con marco rojo, pero no circulares) y otras señales; y en un segundo paso entrenarle para que identifique las distintas señales de velocidad (OPCIÓN 2).

Abstracciones
Como en todo modelo, es necesario plantear la solución asumiendo ciertas abstracciones con el fin de generar resultados y facilitar el avance de las conclusiones. En este caso, las abstracciones serán dos:
•	El programa de conducción del coche asume que el piloto gestiona la conducción de forma autónoma, es decir, esquiva a los distintos coches que puedan estar en la trayectoria o gestiona los cambios de carril necesarios para la conducción.
•	El programa toma las imágenes de entrada directamente con las señales centradas, es decir, en un caso real, habría un software especializado en ubicar la señal en la imagen general de entrada y recortarla para entregársela como input al sistema de procesamiento de imagen; en este caso, se asume que ese paso ya se ha dado y que el sistema recibe una imagen con una señal de tráfico lista para su análisis.
Dadas estas abstracciones, es posible utilizar la función Streetview de Google Maps con el fin de obtener imágenes de las autopistas con el fin de demostrar la hipótesis del proyecto.

Referencias
IEEE	https://ieeexplore.ieee.org/document/8558481
GTSRB	https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
Ppm	    https://github.com/arubiales/Traffic-Signals-TF/blob/master/Models_from_scratch.ipynb
GitHUb	https://github.com/MikeBarkerSpain/MLProyectSpeedSigns

Memoria completa en documentation, título 'Memoria - Proyecto de Machine Learning'

