# graphs.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from regression import sigmoid

"""
@package docstring
Este módulo contiene las funciones necesarias para visualizar los resultados del modelo de regresión logística.

@see https://en.wikipedia.org/wiki/Confusion_matrix
@see https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62
"""

def confusion_matrix(y_true, y_pred):
    """
    Crea una matriz de confusión para los valores verdaderos y predichos.

    @param y_true (array-like): Valores verdaderos.
    @param y_pred (array-like): Valores predichos.
    @return ndarray: Matriz de confusión.
    """
    # Definir las clases
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)))

    for i, true_label in enumerate(classes):
        for j, pred_label in enumerate(classes):
            matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

    return matrix


def plot_confusion_matrix(cm, classes):
    """
    Dibuja una matriz de confusión.

    @param cm (ndarray): Matriz de confusión.
    @param classes (array-like): Etiquetas de las clases.
    """
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))

    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Añadir los valores numéricos a cada celda
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

    plt.show()


def plot_accuracy(test_accuracy, train_accuracy, val_accuracy):
    """
    Dibuja un gráfico de barras para la precisión del conjunto de entrenamiento y prueba.

    @param test_accuracy (float): Precisión en el conjunto de prueba.
    @param train_accuracy (float): Precisión en el conjunto de entrenamiento.
    @param val_accuracy (float): Precisión en el conjunto de validación.
    """
    labels = ['Testing Accuracy', 'Training Accuracy', 'Validation Accuracy']
    accuracies = [test_accuracy, train_accuracy, val_accuracy]

    fig, ax = plt.subplots()
    ax.bar(labels, accuracies, color=['blue', 'orange', 'green'])

    ax.set_ylim(0, 100)
    ax.set_ylabel('Accuracy (%)')

    for i, v in enumerate(accuracies):
        ax.text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom')

    plt.show()


def plot_predicted_probabilities(X, weights):
    """
    Grafica la distribución de las probabilidades predichas.

    @param X_test (ndarray): Matriz de características del conjunto dado.
    @param weights (ndarray): Pesos del modelo.
    @param set (str): Nombre del conjunto de datos.
    """
    # Calcula las probabilidades predichas para el conjunto dado
    z_test = np.dot(X, weights)
    probabilities_test = sigmoid(z_test)

    # Grafica la distribución de las probabilidades predichas
    plt.figure(figsize=(10, 6))
    sns.histplot(probabilities_test, kde=True, bins=30, color='blue')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.show()


def plot_cost(costs):
    """
    Dibuja la evolución del costo durante el entrenamiento.

    @param costs (list): Lista de costos registrados durante cada época.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(costs, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()
