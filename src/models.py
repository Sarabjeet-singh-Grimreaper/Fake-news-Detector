from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def get_models():
    """
    Returns the core VerifiQ ensemble configurations.
    """
    fast_svm = CalibratedClassifierCV(LinearSVC(dual=False, random_state=42), cv=3)
    
    return {
        "KNN": KNeighborsClassifier(
            n_neighbors=9,
            weights="distance",
            n_jobs=-1
        ),
        "LogReg": LogisticRegression(
            C=10.0,
            penalty="l2",
            max_iter=5000,
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
        "SVM": fast_svm,
        "Online_Logistic_Regression": SGDClassifier(
            loss="log_loss",       # Enables predict_proba execution 
            penalty="l2", 
            alpha=0.0001, 
            max_iter=1000, 
            random_state=42
        ),
        "Voting Ensemble": CalibratedClassifierCV(
            estimator=VotingClassifier(
                estimators=[
                    ("logreg", LogisticRegression(C=10.0, penalty="l2", max_iter=5000, random_state=42)),
                    ("rf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
                    ("nn", MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, alpha=0.0001, random_state=42)),
                    ("svm", CalibratedClassifierCV(LinearSVC(dual=False, random_state=42), cv=3))
                ],
                voting="soft"
            ),
            method="sigmoid",
            cv=3
        )
    }