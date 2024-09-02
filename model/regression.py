# regression.py

import numpy as np
import pandas as pd
from model import gradient_descent, split_dataset, predict, compute_cost
from graphs import confusion_matrix, plot_confusion_matrix, plot_accuracy, plot_predicted_probabilities, plot_cost

"""
@package docstring
Este módulo contiene funciones para realizar predicciones con un modelo de regresión logística.

@see https://en.wikipedia.org/wiki/Logistic_regression
@see https://www.beyondphdcoaching.com/dissertation/binary-logistic-regression/
"""

def load_data(file_path):
    """
    Carga los datos transformados desde un archivo y los divide en características y etiquetas.

    @param file_path: Ruta del archivo de datos.
    @return: Tuplas (X, y), donde X son las características y y son las etiquetas.
    """
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    return X, y


def train_and_predict(data_file, learning_rate, epochs, test_size, val_size, random_state):
    """
    Entrena el modelo de regresión logística y realiza predicciones.

    @param data_file: Ruta del archivo con los datos.
    @param learning_rate: Tasa de aprendizaje para el descenso de gradiente.
    @param epochs: Número de épocas para el descenso de gradiente.
    @param test_size: Proporción del dataset a utilizar para pruebas.
    @param random_state: Semilla para el generador de números aleatorios.
    """
    # Cargo los datos
    X, y = load_data(data_file)

    # Divido el dataset en entrenamiento y prueba
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, test_size, val_size, random_state)

    # Agrego una columna de unos para considerar el bias
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    X_val = np.hstack([np.ones((X_val.shape[0], 1)), X_val])

    # Inicializo los pesos
    num_features = X_train.shape[1]
    weights = np.zeros(num_features)

    # Entreno el modelo
    weights, costs = gradient_descent(X_train, y_train, weights, learning_rate, epochs)

    # Realizo predicciones
    test_predictions = predict(X_test, weights)
    train_predictions = predict(X_train, weights)
    val_predictions = predict(X_val, weights)

    # Evalúo mi precisión
    test_accuracy = np.mean(test_predictions == y_test) * 100
    train_accuracy = np.mean(train_predictions == y_train) * 100
    val_accuracy = np.mean(val_predictions == y_val) * 100

    print(f"\nPrecisión en el conjunto de test: {test_accuracy:.2f}%")
    print(f"Precisión en el conjunto de train: {train_accuracy:.2f}%")
    print(f"Precisión en el conjunto de validation: {val_accuracy:.2f}%\n")

    # Evalúo mis costos
    test_cost = compute_cost(X_test, y_test, weights)
    train_cost = compute_cost(X_train, y_train, weights)
    val_cost = compute_cost(X_val, y_val, weights)

    print(f"Costo en el conjunto de test: {test_cost:.4f}")
    print(f"Costo en el conjunto de train: {train_cost:.4f}")
    print(f"Costo en el conjunto de validation: {val_cost:.4f}\n")

    # Evalúo el término final del bias
    bias = weights[0]
    print(f"Término del bias: {bias:.4f}\n")

    # Genero y muestro las matrices de confusión
    cm_test = confusion_matrix(y_test, test_predictions)
    plot_confusion_matrix(cm_test, classes=['Edible', 'Poisonous'])

    cm_train = confusion_matrix(y_train, train_predictions)
    plot_confusion_matrix(cm_train, classes=['Edible', 'Poisonous'])

    cm_val = confusion_matrix(y_val, val_predictions)
    plot_confusion_matrix(cm_val, classes=['Edible', 'Poisonous'])

    # Genero y muestro la gráfica de las probabilidades predichas en test y train
    plot_predicted_probabilities(X_test, weights)
    plot_predicted_probabilities(X_train, weights)
    plot_predicted_probabilities(X_val, weights)

    # Genero y muestro la gráfica de precisión
    plot_accuracy(test_accuracy, train_accuracy, val_accuracy)

    # Genero y muestro la gráfica del costo durante el entrenamiento
    plot_cost(costs)

# Defino mis parámetros
LEARNING_RATE = 0.01
EPOCHS = 10000
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42

# Ejecuto el entrenamiento y la predicción
if __name__ == "__main__":
    train_and_predict(
        'dataset/processed.data', 
        LEARNING_RATE, 
        EPOCHS, 
        TEST_SIZE, 
        VAL_SIZE,
        RANDOM_STATE
    )

