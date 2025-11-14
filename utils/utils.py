# Funciones para el procesamiento de la data

# General

import os
import json
import pandas as pd
import shutil

def tabulate_jsons_from_folder(folder_path):
    '''
    Función para estructurar la información de los metadatos de todas 
    las imágenes.
    
    args
    - folder_path: requiere la ubicación de la carpera que contiene los 
    jsons con los metadatos.
    
    Librerías requeridas:
    - os
    - json
    - pandas as pd 
    '''
    data_list = []

    # Verifica si la carpeta existe
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"La carpeta '{folder_path}' no existe.")

    # Itera sobre los archivos JSON en la carpeta
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                data_list.append(data)

    # Convierte la lista de diccionarios en un DataFrame
    df = pd.DataFrame(data_list)
    return df

# Segformer
from datasets import DatasetDict, Dataset
from PIL import Image
import numpy as np
import os

def load_image_mask_pairs(image_dir, mask_dir, list_id):
    '''
    Función para cargar las imágenes, los labels y procesarlos en 
    el formato necesario para el fine tunning del modelo SegFormer.

    args
    - image_dir: carpeta en la que se encuentran las imágenes.
    - mask_id: carpeta en la que se encuentran los labels.
    - list_id: lista de las observaciones que se van a cargar.

    Librerías requeridas:
    - datasets
    - PIL
    - numpy as np
    - os 
    '''
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    data = []
    for img_file, mask_file in zip(image_files, mask_files):
        #print(img_file)
        #print(img_file.split('.')[0])
        if int(img_file.split('.')[0]) in list_id.to_list():
            img = Image.open(os.path.join(image_dir, img_file)).convert("RGB")
            mask = Image.open(os.path.join(mask_dir, mask_file)).convert("L")
            array = np.array(mask)
            array[array == 255] = 1

            data.append({
                "pixel_values": img,#np.array(img),
                "label": Image.fromarray(array, mode='L')
            })
    return data

# Yolo

def generar_lineas_por_manzana(data):
    '''
    Lee los metadatos del dataset y genera una colección de vértices 
    tal que x_1 y_1 x_2 y_2... . Dicha colección de vértices se 
    encuentra normalizada entre 0 y 1.
    
    args:
    - data: json de metadatos cargado
    '''

    x_max = data["bounds"]['maxx']
    x_min = data["bounds"]['minx']
    y_max = data["bounds"]['maxy']
    y_min = data["bounds"]['miny']

    x_den = x_max - x_min
    y_den = y_max - y_min

    vertices = data["vertices"]
    manzanas = {}

    # Agrupar vértices por idmanzana
    for vert in vertices:
        idm = vert["idmanzana"]
        if idm not in manzanas:
            manzanas[idm] = []
        manzanas[idm].append((vert["x"], vert["y"]))

    # Generar líneas de salida
    lineas = []
    for idm, coords in manzanas.items():
        linea = '0' + " " + " ".join(f"{(x-x_min)/x_den} {1 - (y-y_min)/y_den}" for x, y in coords)
        lineas.append(linea)

    return lineas

def procesar_jsons_en_carpeta(carpeta_entrada, carpeta_salida):
    '''
    Lee de la carpeta de metadatos los jsons con la información de los vértices,
    aplica la función generar_lineas_por_manzana para procesar su información y 
    guarda el resultado compatible con Yolo en txt en otra carpeta. Borra la 
    carpeta de destino para asegurarse de que en la carpeta resultante solo hayan 
    los archivos generados por la última ejecución de la función.

    args:
    - carpeta_entrada: ruta en la que se encuentran los jsons
    - carpeta_salida: ruta en la que se van a guardar los txt

    Librerías requeridas:
    - os
    - json
    - shutil
    '''
    
    # Si la carpeta de salida existe, eliminarla
    if os.path.exists(carpeta_salida):
            shutil.rmtree(carpeta_salida)


    # Crear la carpeta de salida si no existe
    os.makedirs(carpeta_salida)

    for nombre_archivo in os.listdir(carpeta_entrada):
        if nombre_archivo.endswith(".json"):
            ruta_json = os.path.join(carpeta_entrada, nombre_archivo)
            with open(ruta_json, "r", encoding="utf-8") as f:
                datos = json.load(f)
                lineas = generar_lineas_por_manzana(datos)

            # Crear archivo .txt con el mismo nombre que el .json
            nombre_txt = os.path.splitext(nombre_archivo)[0] + ".txt"
            ruta_txt = os.path.join(carpeta_salida, nombre_txt)

            with open(ruta_txt, "w", encoding="utf-8") as f_out:
                for linea in lineas:
                    f_out.write(linea + "\n")

def convertir_vertices_a_yolo_bbox(vertices,x_dim): # pendiente de implementar la normalización mediante el x_dim
    '''
    DEPRECADO
    A partir de la colección de vértices, se genera una caja que los contenga perfectamente.
    '''
    if (len(vertices)) % 2 != 0:
        raise ValueError("La lista de vértices debe tener un número par de elementos (pares x, y).")

    xs = vertices[1::2]
    ys = vertices[2::2]

    x_min = float(min(xs))
    x_max = float(max(xs))
    y_min = float(min(ys))
    y_max = float(max(ys))
    
    # Centro de la caja
    x_center = (x_max - x_min) / 2 + x_min
    y_center = (y_max - y_min) / 2 + y_min

    # Dimensiones de la caja
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height

def convertir_txt_vertices_a_bbox(input_path, output_path, x_path): # pendiente de implementar la normalización mediante el x_path
    try:
        with open(input_path, 'r') as f:
            lines = f.readlines()

        bbox_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3 or len(parts[1:]) % 2 != 0:
                continue  # línea malformada

            class_id = parts[0]
            coords = list(map(float, parts[1:]))
            x_center, y_center, width, height = convertir_vertices_a_yolo_bbox(coords)
            bbox_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        with open(output_path, 'w') as f:
            for line in bbox_lines:
                f.write(line + '\n')
    except Exception as e:
        print(f"Error procesando {input_path}: {e}")

def copiar_archivos_seleccionados(origen, destino, lista_nombres):
    '''
    Función para copiar las observaciones y los labels a las carpetas de train 
    y test para el reentrenamiento de Yolo.

    args:
    - origen: carpeta de origen de los archivos
    - destino: carpeta en la que se van a copiar los archivos
    - lista_nombres: lista de nombres de los archivos que se van a copiar

    Librerías requeridas:
    - shutil 
    - os
    '''
    # Eliminar todos los archivos en la carpeta de destino
    if os.path.exists(destino):
        for archivo in os.listdir(destino):
            ruta_archivo = os.path.join(destino, archivo)
            if os.path.isfile(ruta_archivo):
                os.remove(ruta_archivo)
    else:
        os.makedirs(destino)

    # Copiar archivos seleccionados desde origen a destino
    for nombre in lista_nombres:
        ruta_origen = os.path.join(origen, nombre)
        ruta_destino = os.path.join(destino, nombre)
        #print(ruta_origen,1)
        if os.path.isfile(ruta_origen):
            #print(nombre,2)
            shutil.copy2(ruta_origen, ruta_destino)