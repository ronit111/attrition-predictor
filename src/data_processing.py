"""
Data processing module for employee attrition prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class DataProcessor:
    """Handles data loading, preprocessing, and feature engineering"""

    def __init__(self, data_path='data/HR-Employee-Attrition.csv'):
        self.data_path = data_path
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None

    def load_data(self):
        """Load the dataset"""
        df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        return df

    def engineer_features(self, df):
        """Create additional meaningful features"""
        df = df.copy()

        # Create tenure groups (as numerical)
        df['TenureGroup'] = pd.cut(df['YearsAtCompany'],
                                     bins=[0, 2, 5, 10, 40],
                                     labels=[0, 1, 2, 3])

        # Work-life balance indicator
        df['WorkLifeBalanceScore'] = df['OverTime'].map({'Yes': 0, 'No': 1})

        # Career progression rate
        df['CareerProgressionRate'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)

        # Salary satisfaction (relative to job level)
        df['SalaryToLevel'] = df['MonthlyIncome'] / (df['JobLevel'] + 1)

        return df

    def preprocess(self, df, is_training=True):
        """Preprocess the data for model training/prediction"""
        df = df.copy()

        # Fill NaN values created during feature engineering
        df = df.fillna(0)

        # Drop unnecessary columns
        columns_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        # Separate features and target
        if 'Attrition' in df.columns:
            y = df['Attrition'].map({'Yes': 1, 'No': 0})
            X = df.drop('Attrition', axis=1)
        else:
            y = None
            X = df

        # Handle categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns

        if is_training:
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        else:
            for col in categorical_cols:
                if col in self.label_encoders:
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    X[col] = X[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
                    X[col] = le.transform(X[col].astype(str))

        # Store feature columns
        if is_training:
            self.feature_columns = X.columns.tolist()

        # Scale numerical features
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        return X_scaled, y

    def prepare_for_training(self, test_size=0.2, random_state=42):
        """Complete pipeline for training data"""
        # Load data
        df = self.load_data()

        # Engineer features
        df = self.engineer_features(df)

        # Preprocess
        X, y = self.preprocess(df, is_training=True)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        return X_train, X_test, y_train, y_test, df

    def prepare_single_prediction(self, input_dict):
        """Prepare a single input for prediction"""
        # Convert input dict to dataframe
        df = pd.DataFrame([input_dict])

        # Engineer features
        df = self.engineer_features(df)

        # Preprocess
        X, _ = self.preprocess(df, is_training=False)

        return X
