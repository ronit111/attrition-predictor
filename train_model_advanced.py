"""
Advanced model training script with ensemble learning and optimization
Run this to train the improved model
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processing import DataProcessor
from src.model_advanced import AdvancedAttritionModel
import pandas as pd
import time


def main():
    start_time = time.time()

    print("\n" + "=" * 70)
    print("ADVANCED EMPLOYEE ATTRITION PREDICTOR")
    print("Training with Ensemble Learning & Hyperparameter Optimization")
    print("=" * 70)

    # Initialize data processor
    processor = DataProcessor(data_path='data/HR-Employee-Attrition.csv')

    # Prepare training data
    print("\nLoading and preprocessing data...")
    X_train, X_test, y_train, y_test, df = processor.prepare_for_training()

    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Total features: {X_train.shape[1]}")

    print("\nAttrition Distribution:")
    print(f"No Attrition: {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.2f}%)")
    print(f"Attrition: {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.2f}%)")

    # Initialize and train advanced model
    model = AdvancedAttritionModel()

    # Train with optimization (set optimize=False for faster training)
    # n_trials: higher = better optimization but slower (50-100 recommended)
    print("\n" + "=" * 70)
    print("NOTE: Hyperparameter optimization may take 10-15 minutes")
    print("Set optimize=False in code for faster training (lower accuracy)")
    print("=" * 70)

    model.train_ensemble(
        X_train, y_train,
        use_adasyn=True,
        optimize=True,  # Set to False for faster training
        n_trials=50  # Reduce for faster training (e.g., 20)
    )

    # Evaluate model
    metrics = model.evaluate(X_test, y_test)

    # Show top features
    print("\n" + "=" * 70)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("=" * 70)
    print(model.feature_importance.head(15).to_string(index=False))

    # Compute SHAP values
    print("\n" + "=" * 70)
    print("Computing SHAP values for interpretability...")
    shap_values = model.compute_shap_values(X_test.head(100))
    print("✓ SHAP values computed")

    # Save model
    print("\n" + "=" * 70)
    print("Saving model...")
    model.save_model('models/attrition_model.pkl')

    # Save processor
    print("Saving data processor...")
    import joblib
    joblib.dump(processor, 'models/data_processor.pkl')

    # Final summary
    elapsed_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"✓ Model: Stacking Ensemble (XGBoost + LightGBM + CatBoost)")
    print(f"✓ Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"✓ Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"✓ F1-Score: {metrics['f1_score']:.4f}")
    print(f"✓ CV ROC-AUC: {metrics['cv_scores'].mean():.4f} (+/- {metrics['cv_scores'].std() * 2:.4f})")
    print(f"✓ Total training time: {elapsed_time / 60:.2f} minutes")
    print("=" * 70)

    print("\n🎉 Advanced model training completed successfully!")
    print("You can now run the Streamlit app with: streamlit run app.py")


if __name__ == "__main__":
    main()
