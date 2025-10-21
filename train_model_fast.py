"""
Fast model training script (no optimization) for testing
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processing import DataProcessor
from src.model_advanced import AdvancedAttritionModel
import time


def main():
    start_time = time.time()

    print("\n" + "=" * 70)
    print("FAST TRAINING MODE (No Hyperparameter Optimization)")
    print("=" * 70)

    # Initialize data processor
    processor = DataProcessor(data_path='data/HR-Employee-Attrition.csv')

    # Prepare training data
    print("\nLoading and preprocessing data...")
    X_train, X_test, y_train, y_test, df = processor.prepare_for_training()

    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Total features: {X_train.shape[1]}")

    # Initialize and train model (without optimization)
    model = AdvancedAttritionModel()

    model.train_ensemble(
        X_train, y_train,
        use_adasyn=True,
        optimize=False,  # Fast training
        n_trials=0
    )

    # Evaluate model
    metrics = model.evaluate(X_test, y_test)

    # Show top features
    print("\n" + "=" * 70)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("=" * 70)
    print(model.feature_importance.head(15).to_string(index=False))

    # Compute SHAP values
    print("\nComputing SHAP values...")
    shap_values = model.compute_shap_values(X_test.head(100))

    # Save model
    model.save_model('models/attrition_model.pkl')

    # Save processor
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

    print("\n🎉 Training completed successfully!")


if __name__ == "__main__":
    main()
