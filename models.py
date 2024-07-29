import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
# Load dataset
iris = load_iris()

X, y = iris.data, iris.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Serialize model to a file using pickle
with open('iris_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
