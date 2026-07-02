from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def get_models():
    """
    Initializes and returns the suite of 4 core classifiers 
    configured according to the project guidelines.
    """
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LogReg": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "NeuralNet": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    }
    return models