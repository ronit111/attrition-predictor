"""
Model training and prediction module for employee attrition
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import shap


class AttritionModel:
    """Handles model training, evaluation, and prediction"""

    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.shap_explainer = None
        self.shap_values = None

    def train(self, X_train, y_train, use_smote=True):
        """Train the XGBoost model with optional SMOTE for class imbalance"""

        # Handle class imbalance
        if use_smote:
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train

        # Train XGBoost model
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )

        self.model.fit(X_train_balanced, y_train_balanced)

        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return self.model

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        return metrics

    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        return predictions, probabilities

    def compute_shap_values(self, X_sample):
        """Compute SHAP values for interpretability"""
        # Create SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.model)

        # Calculate SHAP values
        self.shap_values = self.shap_explainer(X_sample)

        return self.shap_values

    def get_prediction_explanation(self, X_single):
        """Get SHAP explanation for a single prediction"""
        if self.shap_explainer is None:
            self.shap_explainer = shap.TreeExplainer(self.model)

        shap_values = self.shap_explainer(X_single)
        return shap_values

    def save_model(self, filepath='models/attrition_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'shap_explainer': self.shap_explainer
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='models/attrition_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_importance = model_data['feature_importance']
        self.shap_explainer = model_data.get('shap_explainer')
        print(f"Model loaded from {filepath}")

    def get_top_risk_factors(self, shap_values, feature_names, top_n=5):
        """Get top N risk factors from SHAP values"""
        # Get absolute SHAP values
        shap_abs = np.abs(shap_values.values[0])

        # Create dataframe with features and their impact
        risk_factors = pd.DataFrame({
            'feature': feature_names,
            'impact': shap_values.values[0],
            'abs_impact': shap_abs
        }).sort_values('abs_impact', ascending=False).head(top_n)

        return risk_factors
