# forest.py

from sklearn.ensemble import RandomForestClassifier

"""
@package docstring
Este módulo contiene funciones para realizar predicciones con un modelo de random forest.

@see https://en.wikipedia.org/wiki/Random_forest
@see https://www.ibm.com/topics/random-forest#:~:text=Random%20forest%20is%20a%20commonly,Decision%20trees
"""

def train_random_forest(X_train, y_train, random_state=42, n_estimators=100):
    """
    Función para entrenar un modelo Random Forest.

    @param X: variables predictoras (features).
    @param y: variable objetivo.
    @param test_size: proporción de datos para el conjunto de prueba.
    @param random_state: semilla para la reproducibilidad.
    @param n_estimators: número de árboles en el bosque.
    @return: modelo entrenado, datos divididos para entrenamiento y prueba.
    """
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    return rf


def predict_forest(model, new_data):
    """
    Función para predecir la clase de nuevos datos usando el modelo entrenado.

    @param rf: modelo Random Forest entrenado.
    @param new_data: datos nuevos para los que se quiere predecir la clase.
    @return: predicciones.
    """
    return model.predict(new_data)