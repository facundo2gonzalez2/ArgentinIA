#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 16:58:38 2022

@author: mcerdeiro
"""
import json
import os
import re
import warnings

# bibliotecas
import cv2
import jamspell
import numpy as np
import pandas as pd
import pytesseract
import math

# herramientas
from src.detect import detect

# directorios
path_in = 'input_data/'
path_out = 'out_data/'


# funciones de preprocesamiento
def preprocesamiento(img) -> np.ndarray:
    # with open(img, 'r') as f:
    m = cv2.imread(img)
    m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return m


# funciones de transcripción
def ocr_tesseract(preprocessed_img):
    """
    Aplica el algoritmo de OCR a la imagen pasada por parámetro.

    :param filename: nombre del archivo.
    :param path_in: directorio donde se encuentra el archivo.
    :return: string con el texto leído de la imagen o None si no pudo procesarse.
    """
    if isinstance(preprocessed_img, np.ndarray):
        custom_config = r'-c tessedit_char_whitelist="AÁBCDEÉFGHIÍJKLMNÑOÓPQRSTUÚVWXYZaábcdeéfghiíjklmnñoópqrstuúvwxyz0123456789 -_/.,:;()"'
        text = str(pytesseract.image_to_string(preprocessed_img, lang="spa", config=custom_config))
        return text

    warnings.warn("Could not process OCR in", path_in )
    return None


# funciones de corrección del texto
def inicializar_corrector(path_modelo='./herramientas/model_juridico.bin'):
    corrector = jamspell.TSpellCorrector()
    assert (corrector.LoadLangModel(path_modelo))
    return corrector


def spellcheck(corrector, texto):
    return corrector.FixFragment(texto)


def unir_saltos_linea(text):
    text = text.replace('-\n', '')
    return text.replace('_\n', '')


def filtrar_simbolosvalidos(texto):
    """
    Reemplaza los símbolos no presentes en el diccionario español por espacios en blanco
    """
    temp = re.sub("[^A-Za-z0-9áéíóúüñÁÉÍÓÚÜÑ.,;:{}()\+\'\"!¡¿?°\[\]\-\s]", ' ', texto)
    # temp = re.sub("\n", ' ', temp)
    return re.sub(' +', ' ', temp)


def es_basura(text):
    """
    Determina como basura si todos las palabras del texto tienen menos de 3 caracteres.
    """
    return all(len(x) < 3 for x in text.split(' '))


def postprocesamiento(text, corrector):
    """
    Transforma el string crudo en texto: une los saltos de línea (unificando las palabras que se cortan al final de la
    línea), filtra los caracteres válidos y aplica el corrector ortográfico.

    :param text: String a postprocesar.
    :param corrector: Corrector ortográfico JamSpell.
    :return: Texto postprocesado.
    """
    text = os.linesep.join([s for s in text.splitlines() if s])
    text = os.linesep.join([s for s in text.splitlines() if not es_basura(s)])
    text = unir_saltos_linea(text)
    # text = spellcheck(corrector, text)
    # text = unir_saltos_linea(text) # por qué otra vez?
    text = filtrar_simbolosvalidos(text)
    # text = text.lower() # por qué?
    text = spellcheck(corrector, text)  # por qué otra vez?
    # text = text.lower()
    return text


# funciones de chequeo de las transcripciones
def palabras_desconocidas(texto, corrector):
    """
    Calcula la cantidad de palabras desconocidas para el corrector del texto.

    :param texto: string.
    :param corrector: Corrector ortográfico JamSpell.
    """
    cantidad = 0
    for palabra in texto.split():
        cantidad += 0 if corrector.WordIsKnown(palabra) else 1
    return cantidad


def encontrar_titulo(bounding_box, titulos, nombre):
    #Tenemos el nombre del bounding box, con esto podemos pensar mejor algunos casos
    x, y, w, h = bounding_box["x"], bounding_box["y"], bounding_box["width"], bounding_box["height"]
    mejor_indice = 0
    max_overlap = -math.inf
    min_distance = math.inf
    overlap_gap = 50
    #La implementación de Overlap hace que un título grande le gane a subtítulos. 
    #Puede ser que esto sea deseado, sino alcanzaría con aumentar el overlap_gap

    #Diferenciamos si es Volanta, ya que en este caso el título que queremos estará por debajo:

    if nombre=="Volanta":
        for i, titulo in enumerate(titulos):
            titulo_x, titulo_y, titulo_w, titulo_h = titulo["x"], titulo["y"], titulo["width"], titulo["height"]
            overlap = max(0, min(x + w, titulo_x + titulo_w) - max(x, titulo_x))
            if (overlap > max_overlap+overlap_gap and titulo_y>y):
                max_overlap = overlap
                mejor_indice = i
                min_distance = math.sqrt((x - titulo_x)**2 + (y - titulo_y)**2)
            elif (max_overlap-overlap_gap <=  overlap) and titulo_y >y and titulo_y<titulos[mejor_indice]["y"]:
                distance = math.sqrt((x - titulo_x)**2 + (y - titulo_y)**2)
                if distance < min_distance:
                    min_distance = distance
                    mejor_indice = i
    else:
        for i, titulo in enumerate(titulos):
            titulo_x, titulo_y, titulo_w, titulo_h = titulo["x"], titulo["y"], titulo["width"], titulo["height"]
            overlap = max(0, min(x + w, titulo_x + titulo_w) - max(x, titulo_x))
            if (overlap > max_overlap+overlap_gap and titulo_y<y):
                max_overlap = overlap
                mejor_indice = i
                min_distance = math.sqrt((x - titulo_x)**2 + (y - titulo_y)**2)
            elif (max_overlap-overlap_gap <=  overlap) and titulo_y <y and titulo_y>titulos[mejor_indice]["y"]:
                distance = math.sqrt((x - titulo_x)**2 + (y - titulo_y)**2)
                if distance < min_distance:
                    min_distance = distance
                    mejor_indice = i

    return mejor_indice



def separar_en_notas(noticia):
    noticia_separada = {
        "Diario": noticia.pop("Diario", {}),
        "Fecha": noticia.pop("Fecha", {}),
        "Notas": []
    }

    bounding_boxes = []
    
    for titulo in noticia.pop("Título", []):
        bounding_boxes.append(titulo["bounding_box"])
        noticia_separada["Notas"].append({
            "Título": [titulo],
        })

    #En noticia separada tenemos las notas y en bounding_boxes los bounding boxes de los titulos

    boundings_titulo = pd.DataFrame(bounding_boxes)
    boundings_titulo["xmin"] = boundings_titulo["x"]
    boundings_titulo["xmax"] = boundings_titulo["x"] + boundings_titulo["width"]
    for nombre, seccion in noticia.items():
        for item in seccion:
            idx = encontrar_titulo(item["bounding_box"], bounding_boxes, nombre)
                # Asignamos el item a la sección correspondiente de la nota
            if noticia_separada["Notas"][idx].get(nombre):
                noticia_separada["Notas"][idx][nombre].append(item)
            else:
                noticia_separada["Notas"][idx][nombre] = [item]

    return noticia_separada


def procesar_imgs(path_in, path_out, modelo_corrector='herramientas/model_juridico.bin'):
    """
    Procesa los archivos '.tif' de 'path_in' con un OCR, aplica un corrector ortográfico, cuenta la proporción de
    palabras desconocidas y los exporta a formato json con toda la noticia como "Cuerpo".
    Además, guarda en 'reporte.csv' la proporción de palabras desconocidas en la imagen.

    :param path_in: directorio de donde tomar las imágenes.
    :param path_out: directorio adonde se guardan los JSON y el CSV de reporte.
    :param modelo_corrector: path del modelo del corrector ortográfico.
    """
    corrector = inicializar_corrector(modelo_corrector)
    data = []
    for filename in os.listdir(path_in):
        if os.path.splitext(filename)[1] == '.tif':
            print(f'Procesando la imagen: {filename}.\n----------')
            preprocessed_img = preprocesamiento(path_in + filename)
            
            bounding_boxes = detect(preprocessed_img)
            noticia_procesada = {}
            
            cant_palabras_desconocidas = 0
            cant_palabras_totales = 0

            for index, bbox in bounding_boxes.iterrows():
                y_min = bbox["ymin"]
                y_max = bbox["ymax"]
                x_min = bbox["xmin"]
                x_max = bbox["xmax"]
                label = bbox["name"]

                imagen = preprocessed_img[int(y_min): int(y_max), int(x_min):int(x_max)]
                texto = ocr_tesseract(imagen)
                texto = postprocesamiento(texto, corrector)
                
                cant_palabras_desconocidas += palabras_desconocidas(texto, corrector)
                cant_palabras_totales += len(texto.split())

                contenido = {
                    "texto": texto,
                    "bounding_box": {
                        "x": x_min,
                        "y": y_min,
                        "width": x_max - x_min,
                        "height": y_max - y_min
                    }
                }
                
                if noticia_procesada.get(label):
                    noticia_procesada[label].append(contenido)
                else:
                    noticia_procesada[label] = [contenido]
            
            pp_desconocidas = cant_palabras_desconocidas / cant_palabras_totales if cant_palabras_totales > 0 else 0
            data.append({'filename': filename, 'palabras_desconocidas': pp_desconocidas})

            notas_separadas = separar_en_notas(noticia_procesada)

            with open(path_out + os.path.splitext(filename)[0] + '.json', "w") as json_file:
                json.dump(notas_separadas, json_file, ensure_ascii=False, indent=4)

    reporte = pd.DataFrame(data=data)
    reporte.to_csv(path_out + 'reporte.csv', index=False)
    print(f'Procesamiento finalizado. Los resultados fueron guardados en el directorio: {path_out}.')
