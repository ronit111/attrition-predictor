"""
Simple model training for cloud deployment (XGBoost only, no ensemble)
Guaranteed compatibility with Streamlit Cloud
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processing import DataProcessor
from src.model import AttritionModel
import joblib


def main():
    print("Training simple XGBoost model for cloud deployment...")

    # Initialize data processor
    processor = DataProcessor(data_path='data/HR-Employee-Attrition.csv')

    # Prepare training data
    X_train, X_test, y_train, y_test, df = processor.prepare_for_training()

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Initialize and train model
    model = AttritionModel()
    model.train(X_train, y_train, use_smote=True)

    # Evaluate
    metrics = model.evaluate(X_test, y_test)

    print(f"\nTest Accuracy: {metrics['roc_auc']:.4f}")

    # Save model
    model.save_model('models/attrition_model.pkl')

    # Save processor
    joblib.dump(processor, 'models/data_processor.pkl')

    print("\n✓ Model training completed!")


if __name__ == "__main__":
    main()
