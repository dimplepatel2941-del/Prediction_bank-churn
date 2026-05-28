# 🏦 Bank Customer Churn Prediction - Streamlit Web Application

A comprehensive Streamlit web application for predicting bank customer churn using a Random Forest machine learning model.

## 📋 Features

- **🔮 Predict Churn**: Input customer data to get real-time churn predictions
- **📊 Data Exploration**: Analyze datasets with interactive visualizations
- **📈 Model Performance**: View model metrics, confusion matrix, and feature importance
- **📤 Batch Predictions**: Upload CSV files to predict churn for multiple customers
- **📊 Dashboard**: Comprehensive metrics and statistics

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (First Time Only)

The model will automatically train on the first run. Alternatively, you can manually train it:

```bash
python train_model.py
```

This will:
- Load training data from `Training data.csv`
- Train a Random Forest model
- Save the model as `churn_model.pkl`

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
├── app.py                    # Main Streamlit application
├── train_model.py            # Model training and evaluation
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
├── Training data.csv         # Training dataset
├── Testing data.csv          # Testing dataset
├── churn_model.pkl          # Trained model (generated after first run)
└── README.md                # This file
```

## 🎯 Usage Guide

### Home Page
- Overview of the application
- Dataset statistics (total customers, churn rate, etc.)

### Predict Churn
- Interactive form to input customer information
- Sliders and checkboxes for all 14 features
- Real-time predictions with probability scores
- Gauge chart visualization of churn probability

### Data Exploration
- View dataset overview
- Feature distributions
- Correlation heatmap
- Statistical summaries
- Churn distribution pie chart

### Model Performance
- Performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- Confusion matrix visualization
- Feature importance ranking
- Detailed classification report

### Upload Data
- Batch predictions on multiple customers
- Download predictions as CSV
- Risk level classification (High/Medium/Low)

## 📊 Model Details

- **Algorithm**: Random Forest Classifier
- **Number of Estimators**: 100
- **Max Depth**: 15
- **Training Data**: ~10,000 customers
- **Features**: 14 (scaled and engineered features)

## 🔧 Features Description

| Feature | Description |
|---------|-------------|
| Scaled Score | Risk/behavior score (scaled) |
| France/Spain | Country indicator (binary) |
| Male | Gender indicator (binary) |
| Scaled Age | Customer age (scaled) |
| Scaled Tenure | Years as customer (scaled) |
| Scaled Balance | Account balance (scaled) |
| NumOfProducts | Number of products held |
| Product to Tenure Ratio | Products per year of tenure |
| IsActiveMember | Active account indicator |
| Scaled Salary | Customer salary (scaled) |
| Balance to Salary Ratio | Financial ratio |
| Engagement Score | Customer engagement level |
| Age Tenure | Interaction feature |

## 📈 Output Format

When making batch predictions, the output CSV includes:
- Original features
- `Churn_Prediction`: 1 = Will churn, 0 = Will retain
- `Churn_Probability`: Probability of churn (0-1)
- `Risk_Level`: High/Medium/Low risk classification

## ⚙️ Configuration

To modify model parameters, edit `train_model.py`:

```python
model = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=15,            # Maximum tree depth
    min_samples_split=10,    # Minimum samples for split
    min_samples_leaf=5,      # Minimum samples in leaf
    random_state=42,         # Random seed
    n_jobs=-1,               # Use all processors
    class_weight='balanced'  # Handle class imbalance
)
```

## 🐛 Troubleshooting

### Model not found error
- Ensure `Training data.csv` exists in the project directory
- The model will automatically train on first run

### Port already in use
```bash
streamlit run app.py --server.port 8502
```

### Out of memory error
- Reduce batch size or use a subset of data

## 📊 Performance Metrics Explanation

- **Accuracy**: Overall correct predictions
- **Precision**: True positives among predicted positives
- **Recall**: True positives among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

## 🔐 Data Privacy

- All predictions are local and not stored
- Downloaded predictions are stored only on your machine
- No data is sent to external servers

## 🎨 Customization

### Change Color Scheme
Edit the CSS in `app.py`:
```python
.churn-risk-high {
    background-color: #ffe6e6;  # Change this color
}
```

### Modify Model Threshold
Update the risk classification in `utils.py` or `app.py`:
```python
df['Risk_Level'] = df['Churn_Probability'].apply(
    lambda x: 'High' if x > 0.6 else ('Medium' if x > 0.3 else 'Low')
)
```

## 📝 Requirements

- Python 3.8+
- Streamlit 1.28.1
- scikit-learn 1.3.2
- pandas 2.1.3
- numpy 1.24.3
- plotly 5.18.0
- seaborn 0.13.0
- joblib 1.3.2
- matplotlib 3.8.2

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all data files are in the correct format
3. Ensure Python dependencies are installed correctly

## 📄 License

This project is created for educational purposes.

---

**Built with ❤️ using Streamlit & Machine Learning**
