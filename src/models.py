from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

def get_models():
    """
    Returns the core VerifiQ ensemble configurations.
    """
    return {
        "KNN": KNeighborsClassifier(
            n_neighbors=9,
            weights="distance",
            n_jobs=-1
        ),
        "LogReg": LogisticRegression(
            C=10.0,
            penalty="l2",
            max_iter=1000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ),
        "NeuralNet": MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=200,
            alpha=0.0001,
            random_state=42
        ),
        "SVM": SVC(
            C=1.0,
            kernel="linear",
            probability=True,
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