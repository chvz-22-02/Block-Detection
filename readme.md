# Proyecto de Detección de Manzanas
Proyecto de analítica avanzanda para predecir la existencia y delimitación de manzanas censales para agilizar la actualización cartográfica realizada por la Dirección Ejecutiva de Cartografía y Geografía (Instituto Nacional de Estadística e Informática).

Para realizar este objetivo se experimentó con redes neuronales de detección de objetos y segmentación de instancias. En particular, se realizó un ejercicio de fine-tunning a los modelos Segformer y Yolo 11 en sus variantes de detección de objetos y OBB (Oriented Bounding Boxes).

En este repositorio se encuentra el conjunto de códigos utilizados para el entrenamiento y validación de los modelos, así como el conjunto de funciones utilizadas para preprocesar y procesar el dataset para su posterior utilización en los distintos modelos.

## Librerías necesarias
Para el correcto funcionamiento de los códigos, se requiere que se tengan instaladas las siguientes librerías y se sugiere las versiones indicadas.
- torch             == 2.9.0+cu130
- datasets          == 4.4.1
- pillow            == 11.3.0
- numpy             == 2.2.6
- pandas            == 2.3.3
- matplotlib        == 3.10.7
- transformers      == 4.57.1
- ultralytics       == 8.3.227
- torch             == 2.9.0+cu130
- torchvision       == 0.24.0+cu130
- evaluate          == 0.4.6
- datasets          == 4.4.1
- PyYAML            == 6.0.3
- typing-inspection == 0.4.2
- typing_extensions == 4.15.0
- matplotlib        == 3.10.7

## Datos utilizados
### Imágenes Satelitales PeruSat-1
Se dispuso de un dataset de imágenes satelitales en formato .tif recogidas por el satélite PeruSat-1 en octubre del 2016 de las ciudades de Chiclayo, Ferreñafe, Lambayeque, Monsefú, Eten y alrededores. Aunque el satélite ofrece imágenes en formato monocromático y multiespectral (ambas ortorrectificadas), se trabajó con la multiespectral.

### Capa de manzanas censales
Se dispuso de la capa oficial de las manzanas censales en formato shp. y complementarios para su uso en GIS, actualizada para su utilización en el Censo de Población y Vivienda del 2017.La geometría de las manzanas se define por el inicio de las vías de tránsito peatonal y/o vehicular como calles, avenidas, caminos o por elementos naturales y/o artificiales como ríos, laderas de cerro, canales, entre otros, de fácil identificación en campo.

## Preprocesamiento de los datos
### Imágenes Satelitales
Se filtraron las regiones superpuestas encontradas en las imágenes, se eliminaron regiones con valores nulos (0 en todas las bandas) y se recortaron las imágenes para poder armar un datase con el cual entrenar los modelos. Debido a que tras el recorte las imágenes resultantes se encontraron en png, se creó de manera paralela un dataset de metadatos para que la información georreferencial no se pierda y pueda ser utilizada.

### Capas
Se corrigió la distorsión generada por la utilización de un sistema métrico distinto a las imágenes satelitales. Dicha corrección se realizó en QGis y la capa resultante fue cortada en las mismas dimensiones que las imágenes satelitales, manteniendo así la superposición con estas y complementando al dataset. Dado que el resultado del recorte fueron matrices, se generó otro dataset de metadatos para mantener la información georreferencial de las capas y sus vértices.

## Procesamiento de los datos
Ya que no se crearon los modelos desde cero, sino que se hizo un fine tunning, se tuvo que adecuar la data para que sea consumible por los códigos de los modelos encontrados así como para realizar la división de los datos en conjuntos de entrenamiento, prueba y validación. Para ello, se utilizaron una serie de funciones contenidas en utils.py.