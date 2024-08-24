# etl.py

import pandas as pd

"""
@package docstring
Este módulo contiene el código para la etapa de extracción, transformación y carga de datos.

@see https://archive.ics.uci.edu/dataset/73/
@see https://en.wikipedia.org/wiki/Extract,_transform,_load
"""

def load_and_transform_data(file_path, output_file_path):
    """
    Carga el dataset, lo transforma a un formato numérico y guarda el DataFrame transformado en un archivo.
    Imputa datos faltantes en el feature 'stalk-root' utilizando la moda de cada clase.

    @param file_path: Ruta del archivo a cargar.
    @param output_file_path: Ruta del archivo a guardar con los datos transformados.
    @return: DataFrame de pandas con los datos transformados.
    """

    # Comienzo definiendo las columnas de mi dataset
    columns = [
        "class", "cap-shape", "cap-surface", "cap-color", "bruises?", "odor",
        "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape",
        "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
        "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
        "ring-number", "ring-type", "spore-print-color", "population", "habitat"
    ]

    # Cargo el dataset en un dataframe de pandas
    df = pd.read_csv(file_path, names=columns)

    # Comienzo transformando los datos a números utilizando un mapeo simple
    mapping = {
        'class': {'e': 0, 'p': 1},
        'cap-shape': {'b': 0, 'c': 1, 'x': 2, 'f': 3, 'k': 4, 's': 5},
        'cap-surface': {'f': 0, 'g': 1, 'y': 2, 's': 3},
        'cap-color': {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'r': 4, 'p': 5, 'u': 6, 'e': 7, 'w': 8, 'y': 9},
        'bruises?': {'t': 0, 'f': 1},
        'odor': {'a': 0, 'l': 1, 'c': 2, 'y': 3, 'f': 4, 'm': 5, 'n': 6, 'p': 7, 's': 8},
        'gill-attachment': {'a': 0, 'd': 1, 'f': 2, 'n': 3},
        'gill-spacing': {'c': 0, 'w': 1, 'd': 2},
        'gill-size': {'b': 0, 'n': 1},
        'gill-color': {'k': 0, 'n': 1, 'b': 2, 'h': 3, 'g': 4, 'r': 5, 'o': 6, 'p': 7, 'u': 8, 'e': 9, 'w': 10, 'y': 11},
        'stalk-shape': {'e': 0, 't': 1},
        'stalk-root': {'b': 0, 'c': 1, 'u': 2, 'e': 3, 'z': 4, 'r': 5, '?': 6},
        'stalk-surface-above-ring': {'f': 0, 'y': 1, 'k': 2, 's': 3},
        'stalk-surface-below-ring': {'f': 0, 'y': 1, 'k': 2, 's': 3},
        'stalk-color-above-ring': {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o': 4, 'p': 5, 'e': 6, 'w': 7, 'y': 8},
        'stalk-color-below-ring': {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o': 4, 'p': 5, 'e': 6, 'w': 7, 'y': 8},
        'veil-type': {'p': 0, 'u': 1},
        'veil-color': {'n': 0, 'o': 1, 'w': 2, 'y': 3},
        'ring-number': {'n': 0, 'o': 1, 't': 2},
        'ring-type': {'c': 0, 'e': 1, 'f': 2, 'l': 3, 'n': 4, 'p': 5, 's': 6, 'z': 7},
        'spore-print-color': {'k': 0, 'n': 1, 'b': 2, 'h': 3, 'r': 4, 'o': 5, 'u': 6, 'w': 7, 'y': 8},
        'population': {'a': 0, 'c': 1, 'n': 2, 's': 3, 'v': 4, 'y': 5},
        'habitat': {'g': 0, 'l': 1, 'm': 2, 'p': 3, 'u': 4, 'w': 5, 'd': 6},
    }

    # Aplico el mapeo utilizando el diccionario anterior
    df_numeric = df.replace(mapping)

    # Según los autores, existen 2480 instancias en el dataset sin valor para el feature 11 (stalk-root), mismas que están denotadas por el símbolo ?, el cual fue mapeado en el paso anterior al número 6.
    # Decidí imputar datos utilizando la moda. Sin embargo, calculo la moda para cada clase con la finalidad de evitar el sesgo que puede introducir esta imputación de datos.

    # Calculo la moda del feature 'stalk-root' para cada clase
    mode_stalk_root_edible = df_numeric[(df_numeric['stalk-root'] != 6) & (df_numeric['class'] == 0)]['stalk-root'].mode()[0]
    mode_stalk_root_poisonous = df_numeric[(df_numeric['stalk-root'] != 6) & (df_numeric['class'] == 1)]['stalk-root'].mode()[0]

    # Imputo los datos faltantes utilizando la moda calculada
    df_numeric.loc[(df_numeric['stalk-root'] == 6) & (df_numeric['class'] == 0), 'stalk-root'] = mode_stalk_root_edible
    df_numeric.loc[(df_numeric['stalk-root'] == 6) & (df_numeric['class'] == 1), 'stalk-root'] = mode_stalk_root_poisonous

    # Guardo el DataFrame transformado en un nuevo archivo
    df_numeric.to_csv(output_file_path, header=False, index=False)

    return df_numeric

# Utilizo mi función para generar un nuevo archivo con los datos transformados
df_numeric = load_and_transform_data('dataset/agaricus-lepiota.data', 'dataset/processed.data')