# regression.py

import numpy as np

"""
@package docstring
Este módulo contiene las funciones de un modelo de regresión logística utilizando la función sigmoide, el descenso de gradiente
y la entropía cruzada binaria como función de pérdida.

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


def gradient_descent(X_train, y_train, X_val, y_val, weights, learning_rate, epochs, patience):
    """
    Realiza el algoritmo de descenso de gradiente para optimizar los pesos,
    incluyendo monitoreo del conjunto de validación y detención temprana.

    @param X_train (ndarray): Matriz de características de entrenamiento.
    @param y_train (ndarray): Matriz de etiquetas de entrenamiento.
    @param X_val (ndarray): Matriz de características de validación.
    @param y_val (ndarray): Matriz de etiquetas de validación.
    @param weights (ndarray): Matriz de pesos.
    @param learning_rate (float): Tasa de aprendizaje para la actualización de los pesos.
    @param epochs (int): Número de iteraciones para el descenso de gradiente.
    @param patience (int): Número de épocas que se esperará antes de detener el entrenamiento si no mejora el conjunto de validación.
    @return tuple: Matriz de pesos optimizados y lista de costos durante el entrenamiento.
    """
    costs_train = []
    costs_val = []
    m = X_train.shape[0]
    
    best_val_cost = float('inf')
    best_weights = None
    patience_counter = 0

    for i in range(epochs):
        z_train = np.dot(X_train, weights)
        h_train = sigmoid(z_train)
        gradient = np.dot(X_train.T, (h_train - y_train)) / m
        weights -= learning_rate * gradient
        
        cost_train = compute_cost(X_train, y_train, weights)
        costs_train.append(cost_train)
        
        cost_val = compute_cost(X_val, y_val, weights)
        costs_val.append(cost_val)
        
        print(f"Epoch {i + 1}, Training Cost: {cost_train:.8f}, Validation Cost: {cost_val:.8f}")

        if cost_val < best_val_cost:
            best_val_cost = cost_val
            best_weights = weights.copy()
            patience_counter = 0 
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Stopping training due to no improvement in validation set.")
            break

    return best_weights, costs_train, costs_val


def split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=None):
    """
    Divide el dataset en conjuntos de entrenamiento, validación y prueba.

    @param X (ndarray): Matriz de características.
    @param y (ndarray): Matriz de etiquetas.
    @param test_size (float): Proporción del dataset a incluir en el conjunto de prueba.
    @param val_size (float): Proporción del conjunto de entrenamiento a incluir en el conjunto de validación.
    @param random_state (int): Semilla para el generador de números aleatorios.
    @return: Tuplas (X_train, X_val, X_test, y_train, y_val, y_test).

    @see https://machinelearningmastery.com/how-to-choose-the-right-test-options-when-evaluating-machine-learning-algorithms/
    @see https://www.v7labs.com/blog/train-validation-test-set
    """
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    test_split_index = int(X.shape[0] * (1 - test_size))
    train_indices = indices[:test_split_index]
    test_indices = indices[test_split_index:]

    val_split_index = int(train_indices.shape[0] * (1 - val_size))
    train_indices_final = train_indices[:val_split_index]
    val_indices = train_indices[val_split_index:]

    X_train = X[train_indices_final]
    X_val = X[val_indices]
    X_test = X[test_indices]
    
    y_train = y[train_indices_final]
    y_val = y[val_indices]
    y_test = y[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test


def predict_regression(X, weights):
    """
    Realiza predicciones utilizando un modelo de regresión logística binaria.

    @param X (ndarray): Matriz de características de las muestras a predecir.
    @param weights (ndarray): Matriz de pesos entrenados del modelo.
    @return ndarray: Un vector de predicciones que contiene la clase predicha para cada muestra.
    """
    linear_model = np.dot(X, weights)
    predictions = sigmoid(linear_model)
    return predictions >= 0.5
