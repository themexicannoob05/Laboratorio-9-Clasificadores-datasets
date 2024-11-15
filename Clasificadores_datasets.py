import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import KFold

# Cargar datasets
iris = load_iris()
wine = load_wine()
cancer = load_breast_cancer()

datasets = {'Iris': iris, 'Wine': wine, 'Breast Cancer': cancer}


# Función para evaluar el clasificador
def evaluate_classifier(X, y, classifier, method):
    if method == 'holdout':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
    elif method == 'crossval':
        scores = cross_val_score(classifier, X, y, cv=10)
        accuracy = np.mean(scores)
        conf_matrix = None  # No se calcula matriz de confusión en cross-validation
    elif method == 'loo':
        loo = LeaveOneOut()
        accuracies = []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
        accuracy = np.mean(accuracies)
        conf_matrix = None  # No se calcula matriz de confusión en LOO
    else:
        raise ValueError("Método no reconocido")

    return accuracy, conf_matrix


# Clasificadores
classifiers = {
    'Euclidiano': KNeighborsClassifier(metric='euclidean'),
    '1NN': KNeighborsClassifier(n_neighbors=1)
}

# Evaluación de los clasificadores con diferentes métodos
for dataset_name, dataset in datasets.items():
    X, y = dataset.data, dataset.target
    print(f"\nEvaluación del dataset {dataset_name}:")

    for method in ['holdout', 'crossval', 'loo']:
        print(f"\nMétodo: {method}")
        for clf_name, clf in classifiers.items():
            print(f"\nClasificador: {clf_name}")
            accuracy, conf_matrix = evaluate_classifier(X, y, clf, method)
            print(f"Accuracy: {accuracy:.4f}")
            if conf_matrix is not None:
                print("Matriz de Confusión:")
                print(conf_matrix)
