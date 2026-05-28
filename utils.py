import pandas as pd
import numpy as np
import joblib
import os
from model import LogisticRegressionModel
import plotly.graph_objects as go
import plotly.express as px


def load_model():
    """Load the trained model."""
    if not os.path.exists('churn_model.pkl'):
        return None

    try:
        model_data = joblib.load('churn_model.pkl')
    except Exception:
        return None

    if isinstance(model_data, dict):
        return LogisticRegressionModel(model_data['weights'], model_data['bias'], model_data['feature_names'])

    if isinstance(model_data, LogisticRegressionModel):
        return model_data

    return None


def load_data(filepath):
    """Load data from CSV or Excel."""
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    else:
        return pd.read_excel(filepath)


def get_feature_names():
    """Get list of feature names for prediction."""
    return [
        'Scaled score', 'France', 'Spain', 'Male', 'Scaled age', 
        'Scaled tenure', 'Scaled balance', 'NumOfProducts',
        'Product to tenure ratio', 'IsActiveMember', 'Scaled salary',
        'Balance to salary ration', 'Engagement score', 'Age tenure'
    ]


def make_prediction(model, input_data):
    """
    Make a single prediction.
    
    Args:
        model: Trained model
        input_data: Dict with feature values
    
    Returns:
        Prediction and probability
    """
    features = get_feature_names()
    X = pd.DataFrame([input_data])[features]

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    return prediction, probability


def get_feature_importance(model, top_n=10):
    """Get feature importance from the model."""
    features = get_feature_names()
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)

    return importance_df


def calculate_statistics(df):
    """Calculate statistics from the dataset."""
    stats = {
        'total_customers': len(df),
        'churned_customers': (df['Exited'] == 1).sum() if 'Exited' in df.columns else 0,
        'retention_rate': 0
    }

    if 'Exited' in df.columns:
        stats['churn_rate'] = (df['Exited'] == 1).sum() / len(df) * 100
        stats['retention_rate'] = (df['Exited'] == 0).sum() / len(df) * 100

    return stats


def create_distribution_chart(df, column, title):
    """Create distribution chart."""
    if column in df.columns:
        fig = px.histogram(df, x=column, nbins=30, title=title,
                          labels={column: column},
                          color_discrete_sequence=['#1f77b4'])
        return fig
    return None


def create_churn_distribution(df):
    """Create churn distribution pie chart."""
    if 'Exited' in df.columns:
        churn_counts = df['Exited'].value_counts()
        labels = ['Retained', 'Churned']
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=churn_counts.values,
            marker=dict(colors=['#2ecc71', '#e74c3c']),
            hoverinfo='label+percent'
        )])
        fig.update_layout(title='Customer Churn Distribution')
        return fig
    return None


def create_correlation_heatmap(df):
    """Create correlation heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0
    ))
    fig.update_layout(title='Feature Correlation Heatmap', height=600, width=700)
    return fig
