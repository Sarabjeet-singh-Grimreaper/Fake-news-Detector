from sklearn.linear_model import SGDClassifier

def get_models():
    """
    Returns the core Veritas ensemble configurations. 
    The 'Online_Logistic_Regression' variant natively supports real-time,
    iterative partial_fit cycles as new worldwide news breaks.
    """
    return {
        "Online_Logistic_Regression": SGDClassifier(
            loss="log_loss",       # Enables predict_proba execution 
            penalty="l2", 
            alpha=0.0001, 
            max_iter=1000, 
            random_state=42
        )
    }