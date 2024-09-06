# tuning.py

import numpy as np
from regression import gradient_descent, predict_regression, split_dataset
from main import load_data
from itertools import product

def grid_search(X_train, y_train, X_val, y_val, learning_rates, epochs_list):
    best_accuracy = 0
    best_params = {}
    
    for lr, ep in product(learning_rates, epochs_list):
        print(f"Testing with Learning Rate={lr}, Epochs={ep}")
        
        # Inicializa los pesos
        num_features = X_train.shape[1]
        weights = np.zeros(num_features)
        
        # Entrena el modelo
        weights, costs_train, costs_val = gradient_descent(X_train, y_train, X_val, y_val, weights, lr, ep, patience=15)
        
        # Realiza predicciones y calcula la precisión
        val_predictions = predict_regression(X_val, weights)
        val_accuracy = np.mean(val_predictions == y_val) * 100
        
        # Guarda los mejores hiperparámetros
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = {'learning_rate': lr, 'epochs': ep}
    
    return best_params, best_accuracy

def main():
    
    X, y = load_data("dataset/processed.data")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42)
    
    # Define los rangos para cada hiperparámetro
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.005, 0.05, 0.003, 0.03]
    epochs_list = [10000, 20000, 30000, 50000]

    # Ejecuta la búsqueda en rejilla
    best_params, best_accuracy = grid_search(X_train, y_train, X_val, y_val, X_test, y_test, learning_rates, epochs_list)

    print(f"Best Accuracy: {best_accuracy:.2f}%")
    print(f"Best Parameters: {best_params}")

if __name__ == "__main__":
    main()