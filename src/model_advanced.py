"""
Advanced model training with ensemble methods and optimization
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import ADASYN, SMOTE
import optuna
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')


class AdvancedAttritionModel:
    """Enhanced model with ensemble learning and hyperparameter optimization"""

    def __init__(self):
        self.best_model = None
        self.ensemble_model = None
        self.feature_importance = None
        self.shap_explainer = None
        self.cv_scores = None

    def optimize_xgboost(self, X_train, y_train, n_trials=100):
        """Optimize XGBoost hyperparameters using Optuna"""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }

            model = XGBClassifier(**params)

            # Stratified K-Fold cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)

            return scores.mean()

        study = optuna.create_study(direction='maximize', study_name='xgboost_optimization')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"\nBest XGBoost ROC-AUC: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")

        return study.best_params

    def optimize_lightgbm(self, X_train, y_train, n_trials=100):
        """Optimize LightGBM hyperparameters using Optuna"""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                'random_state': 42,
                'verbose': -1
            }

            model = LGBMClassifier(**params)

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)

            return scores.mean()

        study = optuna.create_study(direction='maximize', study_name='lightgbm_optimization')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"\nBest LightGBM ROC-AUC: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")

        return study.best_params

    def train_ensemble(self, X_train, y_train, use_adasyn=True, optimize=True, n_trials=50):
        """Train ensemble model with multiple algorithms"""

        print("=" * 70)
        print("ADVANCED MODEL TRAINING")
        print("=" * 70)

        # Handle class imbalance with ADASYN
        if use_adasyn:
            print("\nApplying ADASYN for class balancing...")
            try:
                adasyn = ADASYN(random_state=42, n_neighbors=3)
                X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train, y_train)
                print(f"Original samples: {len(y_train)}, Balanced samples: {len(y_train_balanced)}")
            except Exception as e:
                print(f"ADASYN failed ({str(e)}), falling back to SMOTE...")
                smote = SMOTE(random_state=42)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train

        # Optimize or use default parameters
        if optimize:
            print("\n" + "=" * 70)
            print("HYPERPARAMETER OPTIMIZATION (This may take a while...)")
            print("=" * 70)

            print("\nOptimizing XGBoost...")
            xgb_params = self.optimize_xgboost(X_train_balanced, y_train_balanced, n_trials=n_trials)

            print("\nOptimizing LightGBM...")
            lgb_params = self.optimize_lightgbm(X_train_balanced, y_train_balanced, n_trials=n_trials)

            # CatBoost with default good parameters
            catboost_params = {
                'iterations': 300,
                'depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbose': 0
            }
        else:
            # Use good default parameters
            xgb_params = {
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }

            lgb_params = {
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbose': -1
            }

            catboost_params = {
                'iterations': 300,
                'depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbose': 0
            }

        # Create base models
        print("\n" + "=" * 70)
        print("TRAINING ENSEMBLE MODELS")
        print("=" * 70)

        xgb_model = XGBClassifier(**xgb_params)
        lgb_model = LGBMClassifier(**lgb_params)
        cat_model = CatBoostClassifier(**catboost_params)

        # Create stacking ensemble
        print("\nTraining Stacking Ensemble...")
        self.ensemble_model = StackingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('cat', cat_model)
            ],
            final_estimator=LogisticRegression(random_state=42, max_iter=1000),
            cv=5,
            n_jobs=-1
        )

        self.ensemble_model.fit(X_train_balanced, y_train_balanced)

        # Calibrate probabilities
        print("Calibrating probabilities...")
        self.best_model = CalibratedClassifierCV(self.ensemble_model, method='isotonic', cv=3)
        self.best_model.fit(X_train_balanced, y_train_balanced)

        # Cross-validation scores
        print("\nPerforming 10-Fold Cross-Validation...")
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.ensemble_model, X_train_balanced, y_train_balanced,
                                    cv=skf, scoring='roc_auc', n_jobs=-1)

        self.cv_scores = cv_scores
        print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Calculate feature importance (from fitted estimator within ensemble)
        fitted_xgb = self.ensemble_model.estimators_[0]
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': fitted_xgb.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n✓ Training completed successfully!")

        return self.best_model

    def evaluate(self, X_test, y_test, threshold=0.5):
        """Evaluate model performance with detailed metrics"""

        print("\n" + "=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)

        # Predictions
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)

        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"F1-Score: {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        metrics = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'f1_score': f1,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': cm,
            'cv_scores': self.cv_scores
        }

        return metrics

    def predict(self, X):
        """Make predictions"""
        predictions = self.best_model.predict(X)
        probabilities = self.best_model.predict_proba(X)[:, 1]
        return predictions, probabilities

    def compute_shap_values(self, X_sample):
        """Compute SHAP values for interpretability"""
        # Use the XGBoost model from ensemble for SHAP
        xgb_model = self.ensemble_model.estimators_[0]
        self.shap_explainer = shap.TreeExplainer(xgb_model)
        self.shap_values = self.shap_explainer(X_sample)
        return self.shap_values

    def get_prediction_explanation(self, X_single):
        """Get SHAP explanation for a single prediction"""
        if self.shap_explainer is None:
            xgb_model = self.ensemble_model.estimators_[0]
            self.shap_explainer = shap.TreeExplainer(xgb_model)

        shap_values = self.shap_explainer(X_single)
        return shap_values

    def save_model(self, filepath='models/attrition_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.best_model,
            'ensemble': self.ensemble_model,
            'feature_importance': self.feature_importance,
            'shap_explainer': self.shap_explainer,
            'cv_scores': self.cv_scores
        }
        joblib.dump(model_data, filepath)
        print(f"\n✓ Model saved to {filepath}")

    def load_model(self, filepath='models/attrition_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.ensemble_model = model_data.get('ensemble')
        self.feature_importance = model_data['feature_importance']
        self.shap_explainer = model_data.get('shap_explainer')
        self.cv_scores = model_data.get('cv_scores')
        print(f"Model loaded from {filepath}")

    def get_top_risk_factors(self, shap_values, feature_names, top_n=5):
        """Get top N risk factors from SHAP values"""
        shap_abs = np.abs(shap_values.values[0])

        risk_factors = pd.DataFrame({
            'feature': feature_names,
            'impact': shap_values.values[0],
            'abs_impact': shap_abs
        }).sort_values('abs_impact', ascending=False).head(top_n)

        return risk_factors
