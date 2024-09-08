# tuning.py

import numpy as np
from regression import gradient_descent, predict_regression, split_dataset
from main import load_data

def calculate_accuracy(X, y, weights):
    predictions = predict_regression(X, weights)
    accuracy = np.mean(predictions == y)
    return accuracy

def main():
    
    X, y = load_data("dataset/processed.data")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42)
    
    learning_rates = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1]
    epochs = 10000
    results = []
    patience = 10

    for lr in learning_rates:
        weights = np.zeros(X_train.shape[1])

        print(f"\nTraining with learning rate {lr}...")
        weights, costs_train, costs_val = gradient_descent(X_train, y_train, X_val, y_val, weights, learning_rate=lr, epochs=epochs, patience=patience)
        
        final_val_cost = costs_val[-1]
        accuracy_val = calculate_accuracy(X_val, y_val, weights)
        results.append((lr, final_val_cost, accuracy_val))

    print("\nResultados finales:")
    for lr, cost, accuracy in results:
        print(f"Learning rate {lr} - Validation Cost: {cost:.8f}, Validation Accuracy: {accuracy:.4f}")
    print("")

if __name__ == "__main__":
    main()