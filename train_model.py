import os
import joblib
import pandas as pd
import numpy as np
from model import LogisticRegressionModel


def train_logistic_regression(X, y, learning_rate=0.1, epochs=5000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0

    for _ in range(epochs):
        linear_model = X.dot(weights) + bias
        predictions = 1 / (1 + np.exp(-linear_model))

        dw = (1 / n_samples) * X.T.dot(predictions - y)
        db = (1 / n_samples) * np.sum(predictions - y)

        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias


def roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    desc = np.argsort(-y_score)
    y_true = y_true[desc]
    y_score = y_score[desc]

    distinct_values = np.unique(y_score)
    thresholds = np.r_[distinct_values, distinct_values[-1] - 1e-9]

    tpr = []
    fpr = []
    for thr in thresholds:
        preds = y_score >= thr
        tp = np.sum((y_true == 1) & preds)
        fp = np.sum((y_true == 0) & preds)
        fn = np.sum((y_true == 1) & ~preds)
        tn = np.sum((y_true == 0) & ~preds)

        tpr.append(tp / (tp + fn) if tp + fn > 0 else 0.0)
        fpr.append(fp / (fp + tn) if fp + tn > 0 else 0.0)

    tpr = np.array(tpr)
    fpr = np.array(fpr)
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2
    return auc


def classification_report(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    report = []
    for label in [0, 1]:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        support = np.sum(y_true == label)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        report.append(
            f"{label}       {precision:.2f}      {recall:.2f}      {f1:.2f}      {support}"
        )

    accuracy = np.mean(y_true == y_pred)
    report_text = (
        "              precision    recall  f1-score   support\n"
        + "\n".join(report)
        + f"\n\naccuracy    {accuracy:.2f}\n"
    )
    return report_text
def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])


def train_model(data_path='Training data.csv'):
    df = pd.read_csv(data_path)
    X = df.drop('Exited', axis=1)
    y = df['Exited'].astype(int).values

    weights, bias = train_logistic_regression(X.values, y, learning_rate=0.05, epochs=8000)
    model = LogisticRegressionModel(weights, bias, X.columns)

    model_data = {
        'weights': weights,
        'bias': bias,
        'feature_names': X.columns.tolist()
    }
    joblib.dump(model_data, 'churn_model.pkl')
    print("✓ Model trained and saved successfully!")

    metrics = evaluate_model(model, X.values, y)
    return model, X.columns.tolist(), metrics


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    tp = np.sum((y_test == 1) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_test)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }
    return metrics


if __name__ == "__main__":
    train_model()
