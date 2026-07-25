from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def get_models():
    """
    Returns the V2.0 production models:
    - Logistic Regression (highly interpretable baseline)
    - Random Forest (non-linear representation)
    - Linear SVM (calibrated margin classifier)
    - Voting Ensemble (consensus classifier of the above three)
    """
    logreg = LogisticRegression(
        C=1.0,
        penalty="l2",
        max_iter=2000,
        random_state=42
    )
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    
    svm = CalibratedClassifierCV(
        estimator=LinearSVC(dual=False, random_state=42),
        cv=3
    )
    
    voting_ensemble = VotingClassifier(
        estimators=[
            ("logreg", logreg),
            ("rf", rf),
            ("svm", svm)
        ],
        voting="soft"
    )
    
    return {
        "LogReg": logreg,
        "Random Forest": rf,
        "SVM": svm,
        "Voting Ensemble": voting_ensemble
    }