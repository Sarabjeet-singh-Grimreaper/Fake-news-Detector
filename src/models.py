from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def get_models():
    """
    Returns the core VerifiQ ensemble configurations.
    """
    return {
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=-1
        ),
        "LogReg": LogisticRegression(
            C=1.0,
            penalty="l2",
            max_iter=1000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=30,
            random_state=42,
            n_jobs=-1
        ),
        "NeuralNet": MLPClassifier(
            hidden_layer_sizes=(50,),
            max_iter=100,
            alpha=0.0001,
            random_state=42
        ),
        "Online_Logistic_Regression": SGDClassifier(
            loss="log_loss",       # Enables predict_proba execution 
            penalty="l2", 
            alpha=0.0001, 
            max_iter=1000, 
            random_state=42
        )
    }