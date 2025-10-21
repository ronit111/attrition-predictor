"""
Script to train the attrition prediction model
Run this to create the model file for the Streamlit app
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processing import DataProcessor
from src.model import AttritionModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def perform_eda(df):
    """Perform basic exploratory data analysis"""
    print("=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    print("\nDataset Shape:", df.shape)
    print("\nAttrition Distribution:")
    print(df['Attrition'].value_counts())
    print("\nAttrition Rate:", df['Attrition'].value_counts(normalize=True)['Yes'] * 100, "%")

    print("\n" + "=" * 50)


def main():
    print("Starting model training pipeline...\n")

    # Initialize data processor
    processor = DataProcessor(data_path='data/HR-Employee-Attrition.csv')

    # Prepare training data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, df = processor.prepare_for_training()

    # Perform EDA
    perform_eda(df)

    # Initialize and train model
    print("\nTraining XGBoost model...")
    model = AttritionModel()
    model.train(X_train, y_train, use_smote=True)

    # Evaluate model
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test, y_test)

    print(f"\nROC-AUC Score: {metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])

    # Show feature importance
    print("\nTop 10 Important Features:")
    print(model.feature_importance.head(10))

    # Compute SHAP values for test set sample
    print("\nComputing SHAP values...")
    shap_values = model.compute_shap_values(X_test.head(100))

    # Save model
    print("\nSaving model...")
    model.save_model('models/attrition_model.pkl')

    # Save processor
    print("Saving data processor...")
    import joblib
    joblib.dump(processor, 'models/data_processor.pkl')

    print("\n" + "=" * 50)
    print("✓ Model training completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
