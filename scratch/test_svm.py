import os
import pickle
from src.features import extract_features
from src.models import get_models

X_train, X_test, y_train, y_test, vectorizer = extract_features()
models = get_models()
svm = models["SVM"]
print("Fitting SVM...")
svm.fit(X_train[:100], y_train[:100])
print("Fitted. Saving...")
with open("models/svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)
print("Saved. Checking existence...")
print("Exists:", os.path.exists("models/svm_model.pkl"))
