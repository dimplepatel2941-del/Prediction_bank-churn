import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import os
from train_model import train_model, evaluate_model
from utils import (
    load_model, load_data, get_feature_names, make_prediction,
    get_feature_importance, calculate_statistics, create_distribution_chart,
    create_churn_distribution, create_correlation_heatmap
)

# Page configuration
st.set_page_config(
    page_title="Bank Customer Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .churn-risk-high {
        background-color: #ffe6e6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #e74c3c;
    }
    .churn-risk-low {
        background-color: #e6ffe6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2ecc71;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'features' not in st.session_state:
    st.session_state.features = None

# Sidebar
st.sidebar.title("🏦 Bank Churn Predictor")
page = st.sidebar.radio("Select Page", 
    ["Home", "Predict Churn", "Data Exploration", "Model Performance", "Upload Data"])

# Load or train model
if st.session_state.model is None:
    if os.path.exists('churn_model.pkl'):
        st.session_state.model = load_model()
        if st.session_state.model is None:
            os.remove('churn_model.pkl')
        else:
            st.session_state.features = get_feature_names()
    if st.session_state.model is None:
        st.info("🔧 Model not found. Training model from data...")
        try:
            model, features, metrics = train_model('Training data.csv')
            st.session_state.model = model
            st.session_state.features = features
            st.success("✓ Model trained successfully!")
        except ImportError as e:
            st.error(
                "scikit-learn is not installed in this environment.\n"
                "Install it with: `pip install scikit-learn` and rerun the app,\n"
                "or run the training script locally using the included `run_app.py`."
            )
        except Exception as e:
            st.error(f"Error training model: {e}")

# Home Page
if page == "Home":
    st.title("🏦 Bank Customer Churn Prediction")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 What This App Does
        - **Predict Customer Churn**: Identify customers at risk of leaving
        - **Analyze Patterns**: Explore data and discover trends
        - **Model Insights**: View model performance and feature importance
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Key Features
        - Real-time predictions
        - Interactive visualizations
        - Model performance metrics
        - Feature importance analysis
        """)
    
    with col3:
        st.markdown("""
        ### 📈 Getting Started
        1. Go to **Predict Churn** for predictions
        2. Check **Data Exploration** for analysis
        3. Review **Model Performance** for metrics
        4. Upload new data if needed
        """)
    
    st.markdown("---")
    
    # Load and display dataset statistics
    try:
        train_data = load_data('Training data.csv')
        stats = calculate_statistics(train_data)
        
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", stats['total_customers'])
        with col2:
            st.metric("Churned Customers", stats['churned_customers'])
        with col3:
            st.metric("Churn Rate", f"{stats['churn_rate']:.2f}%")
        with col4:
            st.metric("Retention Rate", f"{stats['retention_rate']:.2f}%")
    except Exception as e:
        st.error(f"Error loading data: {e}")

# Prediction Page
elif page == "Predict Churn":
    st.title("🔮 Predict Customer Churn")
    st.markdown("---")
    
    if st.session_state.model is None:
        st.error("Model not loaded. Please train the model first.")
    else:
        st.subheader("Enter Customer Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scaled_score = st.slider("Scaled Score", -3.0, 3.0, 0.0, step=0.1)
            france = st.checkbox("France", value=True)
            spain = st.checkbox("Spain", value=False)
            male = st.checkbox("Male", value=True)
            scaled_age = st.slider("Scaled Age", -2.0, 3.0, 0.0, step=0.1)
            scaled_tenure = st.slider("Scaled Tenure", -1.5, 2.5, 0.0, step=0.1)
            scaled_balance = st.slider("Scaled Balance", -1.5, 3.0, 0.0, step=0.1)
        
        with col2:
            num_products = st.slider("Number of Products", 1, 4, 2)
            product_tenure_ratio = st.slider("Product to Tenure Ratio", 0.0, 2.0, 0.5, step=0.05)
            is_active = st.checkbox("Is Active Member", value=True)
            scaled_salary = st.slider("Scaled Salary", -1.0, 3.0, 0.5, step=0.1)
            balance_salary_ratio = st.slider("Balance to Salary Ratio", -2.0, 3.0, 0.5, step=0.1)
            engagement_score = st.slider("Engagement Score", 0.0, 1.0, 0.5, step=0.1)
            age_tenure = st.slider("Age Tenure Interaction", 0.0, 1.0, 0.5, step=0.05)
        
        # Create input data
        input_data = {
            'Scaled score': scaled_score,
            'France': int(france),
            'Spain': int(spain),
            'Male': int(male),
            'Scaled age': scaled_age,
            'Scaled tenure': scaled_tenure,
            'Scaled balance': scaled_balance,
            'NumOfProducts': num_products,
            'Product to tenure ratio': product_tenure_ratio,
            'IsActiveMember': int(is_active),
            'Scaled salary': scaled_salary,
            'Balance to salary ration': balance_salary_ratio,
            'Engagement score': engagement_score,
            'Age tenure': age_tenure
        }
        
        if st.button("🎯 Make Prediction", key="predict_btn"):
            try:
                prediction, probability = make_prediction(st.session_state.model, input_data)
                
                st.markdown("---")
                st.subheader("📊 Prediction Result")
                
                churn_prob = probability[1] * 100
                retain_prob = probability[0] * 100
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="churn-risk-high">
                    <h3>⚠️ HIGH RISK - Customer Likely to Churn</h3>
                    <p><strong>Churn Probability: {churn_prob:.2f}%</strong></p>
                    <p><strong>Retention Probability: {retain_prob:.2f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="churn-risk-low">
                    <h3>✓ LOW RISK - Customer Likely to Retain</h3>
                    <p><strong>Churn Probability: {churn_prob:.2f}%</strong></p>
                    <p><strong>Retention Probability: {retain_prob:.2f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability gauge
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_gauge = {
                        'data': [{
                            'type': 'indicator',
                            'mode': 'gauge+number+delta',
                            'value': churn_prob,
                            'domain': {'x': [0, 1], 'y': [0, 1]},
                            'title': {'text': 'Churn Probability (%)'},
                            'gauge': {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': 'darkblue'},
                                'steps': [
                                    {'range': [0, 33], 'color': '#2ecc71'},
                                    {'range': [33, 66], 'color': '#f39c12'},
                                    {'range': [66, 100], 'color': '#e74c3c'}
                                ],
                                'threshold': {
                                    'line': {'color': 'red', 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        }],
                        'layout': {'height': 400}
                    }
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")

# Data Exploration Page
elif page == "Data Exploration":
    st.title("📊 Data Exploration & Analysis")
    st.markdown("---")
    
    try:
        data = load_data('Training data.csv')
        
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rows", len(data))
        with col2:
            st.metric("Columns", len(data.columns))
        with col3:
            st.metric("Missing Values", data.isnull().sum().sum())
        
        st.markdown("---")
        
        # Display data sample
        with st.expander("View Dataset Sample"):
            st.dataframe(data.head(10), use_container_width=True)
        
        st.markdown("---")
        
        # Distributions
        st.subheader("📈 Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn distribution
            fig_churn = create_churn_distribution(data)
            if fig_churn:
                st.plotly_chart(fig_churn, use_container_width=True)
        
        with col2:
            # Select a feature to plot
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            selected_col = st.selectbox("Select Feature to Explore", numeric_cols)
            
            fig = create_distribution_chart(data, selected_col, f"Distribution of {selected_col}")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Correlation heatmap
        st.subheader("🔗 Feature Correlations")
        fig_corr = create_correlation_heatmap(data)
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("---")
        
        # Statistical summary
        st.subheader("📊 Statistical Summary")
        st.dataframe(data.describe(), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")

# Model Performance Page
elif page == "Model Performance":
    st.title("📈 Model Performance")
    st.markdown("---")
    
    if st.session_state.model is None:
        st.error("Model not loaded.")
    else:
        try:
            # Load test data
            test_data = load_data('Testing data.csv')
            X_test = test_data.drop('Exited', axis=1)
            y_test = test_data['Exited']
            
            # Evaluate model
            metrics = evaluate_model(st.session_state.model, X_test, y_test)
            
            st.subheader("🎯 Performance Metrics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{metrics['f1']:.4f}")
            with col5:
                st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            
            st.markdown("---")
            
            # Confusion Matrix
            st.subheader("🔍 Confusion Matrix")
            cm = metrics['confusion_matrix']
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted: No Churn', 'Predicted: Churn'],
                y=['Actual: No Churn', 'Actual: Churn'],
                text=cm,
                texttemplate='%{text}',
                colorscale='Blues'
            ))
            fig_cm.update_layout(title='Confusion Matrix', height=400)
            st.plotly_chart(fig_cm, use_container_width=True)
            
            st.markdown("---")
            
            # Feature Importance
            st.subheader("⭐ Feature Importance")
            importance_df = get_feature_importance(st.session_state.model, top_n=10)
            
            fig_importance = go.Figure(data=[
                go.Bar(x=importance_df['Importance'], 
                       y=importance_df['Feature'],
                       orientation='h',
                       marker_color='#3498db')
            ])
            fig_importance.update_layout(
                title='Top 10 Most Important Features',
                xaxis_title='Importance',
                yaxis_title='Feature',
                height=500
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
            st.markdown("---")
            
            # Classification Report
            st.subheader("📋 Classification Report")
            st.text(metrics['classification_report'])
            
        except FileNotFoundError as e:
            st.warning("Testing data not found. Please ensure 'Testing data.csv' exists in the project directory.")
        except ImportError as e:
            st.error(
                "scikit-learn is not installed in this environment.\n"
                "Install it with: `pip install scikit-learn` and rerun the app,\n"
                "or evaluate the model locally after installing dependencies."
            )
        except Exception as e:
            st.error(f"Error evaluating model: {e}")

# Upload Data Page
elif page == "Upload Data":
    st.title("📤 Upload New Data")
    st.markdown("---")
    
    st.info("Upload a CSV file with the same format as the training data to make predictions on multiple customers.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.subheader("Uploaded Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🔮 Predict for All Rows"):
                if st.session_state.model is not None:
                    predictions = []
                    probabilities = []
                    
                    progress_bar = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        input_dict = row.to_dict()
                        pred, prob = make_prediction(st.session_state.model, input_dict)
                        predictions.append(pred)
                        probabilities.append(prob[1])
                        
                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)
                    
                    # Add predictions to dataframe
                    df['Churn_Prediction'] = predictions
                    df['Churn_Probability'] = probabilities
                    df['Risk_Level'] = df['Churn_Probability'].apply(
                        lambda x: 'High' if x > 0.6 else ('Medium' if x > 0.3 else 'Low')
                    )
                    
                    st.subheader("Predictions")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    st.markdown("---")
                    st.subheader("📊 Prediction Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        high_risk = (df['Risk_Level'] == 'High').sum()
                        st.metric("High Risk Customers", high_risk)
                    
                    with col2:
                        medium_risk = (df['Risk_Level'] == 'Medium').sum()
                        st.metric("Medium Risk Customers", medium_risk)
                    
                    with col3:
                        low_risk = (df['Risk_Level'] == 'Low').sum()
                        st.metric("Low Risk Customers", low_risk)
                else:
                    st.error("Model not loaded.")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    🏦 Bank Customer Churn Prediction System | Built with Streamlit & Random Forest
</div>
""", unsafe_allow_html=True)
