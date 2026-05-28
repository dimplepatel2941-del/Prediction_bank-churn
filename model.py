import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegressionModel:
    def __init__(self, weights, bias, feature_names):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)
        self.feature_names = list(feature_names)

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=float)
        logits = X_arr.dot(self.weights) + self.bias
        probs = sigmoid(logits)
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        return np.abs(self.weights)


def build_model_from_weights(data):
    weights = data['weights']
    bias = data['bias']
    features = data['feature_names']
    return LogisticRegressionModel(weights, bias, features)
