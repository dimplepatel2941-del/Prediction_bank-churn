#!/usr/bin/env python
"""
Quick startup script for the Bank Customer Churn Prediction application.
This script handles initial setup and runs the Streamlit app.
"""

import sys
import subprocess
import os

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]} detected")

def install_dependencies():
    """Install required dependencies from requirements.txt."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies: {e}")
        sys.exit(1)

def train_model_if_needed():
    """Train model if it doesn't exist."""
    if not os.path.exists('churn_model.pkl'):
        print("\n🔧 Model not found. Training model...")
        try:
            from train_model import train_model
            model, features, metrics = train_model('Training data.csv')
            print("✓ Model trained successfully!")
            print(f"  - Accuracy: {metrics['accuracy']:.4f}")
            print(f"  - Precision: {metrics['precision']:.4f}")
            print(f"  - Recall: {metrics['recall']:.4f}")
            print(f"  - F1-Score: {metrics['f1']:.4f}")
        except Exception as e:
            print(f"ERROR: Failed to train model: {e}")
            sys.exit(1)
    else:
        print("\n✓ Model found. Skipping training.")

def run_streamlit_app():
    """Run the Streamlit application."""
    print("\n🚀 Starting Streamlit application...")
    print("   The app will open at http://localhost:8501")
    print("   Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n✓ Application stopped")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Failed to run Streamlit: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    print("=" * 50)
    print("🏦 Bank Customer Churn Prediction")
    print("=" * 50)
    
    check_python_version()
    install_dependencies()
    train_model_if_needed()
    run_streamlit_app()

if __name__ == "__main__":
    main()
