# model.py

import numpy as np

"""
@package docstring
Este módulo implementa un modelo de regresión logística utilizando la función sigmoide y el descenso de gradiente para la clasificación.

@see https://en.wikipedia.org/wiki/Logistic_regression
@see https://www.beyondphdcoaching.com/dissertation/binary-logistic-regression/
"""

def sigmoid(z):
    """
    Calcula la función sigmoide para la entrada z.

    @param z (ndarray): Entrada para la función sigmoide.
    @return h ndarray: Resultados de aplicar la función sigmoide a la entrada z.

    @see https://en.wikipedia.org/wiki/Sigmoid_function
    """
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, weights):
    """
    Calcula el costo (función de pérdida) de la regresión logística.
    Mide qué tan bien el modelo se ajusta a los datos de entrenamiento.
    Al ser una regresión logística binaria, el costo es la entropía cruzada binaria.

        @param X (ndarray): Matriz de características.
        @param y (ndarray): Vector de etiquetas para la clase específica.
        @param weights (ndarray): Vector de pesos para la clase específica.
        @return float: Valor del costo calculado.

        @see https://en.wikipedia.org/wiki/Cross_entropy
        @see https://machinelearningmastery.com/cross-entropy-for-machine-learning/
        @see https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
        @see https://www.analyticsvidhya.com/blog/2020/11/binary-cross-entropy-aka-log-loss-the-cost-function-used-in-logistic-regression/
    """
    m = X.shape[0]
    z = np.dot(X, weights)
    h = sigmoid(z)
    cost = -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / m
    return cost


def gradient_descent(X, y, weights, learning_rate, epochs):
    """
    Realiza el algoritmo de descenso de gradiente para optimizar los pesos.

        @param X (ndarray): Matriz de características.
        @param y (ndarray): Matriz de etiquetas en formato one-hot.
        @param weights (ndarray): Matriz de pesos.
        @param learning_rate (float): Tasa de aprendizaje para la actualización de los pesos.
        @param epochs (int): Número de iteraciones para el descenso de gradiente.
        @return ndarray: Matriz de pesos optimizados después de aplicar el descenso de gradiente.

        @see https://en.wikipedia.org/wiki/Gradient_descent
        @see https://www.geeksforgeeks.org/how-to-implement-a-gradient-descent-in-python-to-find-a-local-minimum/
        @see https://induraj2020.medium.com/implementing-gradient-descent-in-python-d1c6aeb9a448
        @see https://www.youtube.com/watch?v=IHZwWFHWa-w
        @see https://www.youtube.com/watch?v=sDv4f4s2SB8
    """
    costs = []
    m = X.shape[0]

    for i in range(epochs):
        z = np.dot(X, weights)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / m
        weights -= learning_rate * gradient
        cost = compute_cost(X, y, weights)
        costs.append(cost)
        print(f"Epoch {i + 1}, Cost: {cost}")

    return weights, costs


def split_dataset(X, y, test_size=0.2, random_state=None):
    """
    Divide el dataset en conjuntos de entrenamiento y prueba.

    @param X (ndarray): Matriz de características.
    @param y (ndarray): Matriz de etiquetas.
    @param test_size (float): Proporción del dataset a incluir en el conjunto de prueba.
    @param random_state (int): Semilla para el generador de números aleatorios.
    @return: Tuplas (X_train, X_test, y_train, y_test).

    @see https://machinelearningmastery.com/how-to-choose-the-right-test-options-when-evaluating-machine-learning-algorithms/
    @see https://www.v7labs.com/blog/train-validation-test-set
    """
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    split_index = int(X.shape[0] * (1 - test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def predict(X, weights):
    """
    Realiza predicciones utilizando un modelo de regresión logística binaria.

    @param X (ndarray): Matriz de características de las muestras a predecir.
    @param weights (ndarray): Matriz de pesos entrenados del modelo.
    @return ndarray: Un vector de predicciones que contiene la clase predicha para cada muestra.
    """
    linear_model = np.dot(X, weights)
    predictions = sigmoid(linear_model)
    return predictions >= 0.5
