"""
Model Training Module
Phase 2: Training ML models for risk classification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import xgboost as xgb
from lightgbm import LGBMClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Trains and evaluates machine learning models for health risk classification."""
    
    def __init__(self, feature_data_path: str):
        """
        Initialize the model trainer.
        
        Args:
            feature_data_path: Path to the feature-engineered dataset
        """
        self.feature_data_path = feature_data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.predictions = {}
        self.performance = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load the feature-engineered dataset."""
        print(f"Loading feature data from {self.feature_data_path}...")
        self.df = pd.read_csv(self.feature_data_path)
        print(f"Dataset shape: {self.df.shape}")
        return self.df
    
    def prepare_features(self) -> tuple:
        """
        Prepare features and labels for modeling.
        Excludes demographic features (Age, Height, Weight, BMI, Gender).
        """
        print("Preparing features...")
        
        # One-hot encode categorical features
        categorical_features = ['gender']
        df_encoded = pd.get_dummies(self.df, columns=categorical_features, drop_first=True)
        
        # Define features to exclude
        exclude_features = [
            'user_id', 'day', 'Anomaly', 'Risk_Level', 'Risk_Level_Cluster',
            'Age (years)', 'Height (meter)', 'Weight (kg)', 'BMI'
        ]
        
        # Select features (exclude demographics)
        feature_cols = [col for col in df_encoded.columns if col not in exclude_features]
        
        X = df_encoded[feature_cols]
        y = df_encoded['Risk_Level_Cluster']
        
        print(f"✅ Selected {len(feature_cols)} features (excluding demographics)")
        print(f"Target distribution:\n{y.value_counts()}")
        
        return X, y, feature_cols
    
    def split_data(self, X, y, test_size: float = 0.2, random_state: int = 42):
        """Split data into training and testing sets."""
        print(f"Splitting data (test_size={test_size})...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Testing set: {self.X_test.shape[0]} samples")
    
    def train_random_forest(self):
        """Train Random Forest model."""
        print("\n[1/4] Training Random Forest...")
        
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(self.X_train, self.y_train)
        y_pred = rf_model.predict(self.X_test)
        
        self.models['Random Forest'] = rf_model
        self.predictions['Random Forest'] = y_pred
        
        self._evaluate_model('Random Forest', y_pred)
        
        return rf_model
    
    def train_xgboost(self):
        """Train XGBoost model."""
        print("\n[2/4] Training XGBoost...")
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softmax',
            num_class=3,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        
        xgb_model.fit(self.X_train, self.y_train)
        y_pred = xgb_model.predict(self.X_test)
        
        self.models['XGBoost'] = xgb_model
        self.predictions['XGBoost'] = y_pred
        
        self._evaluate_model('XGBoost', y_pred)
        
        return xgb_model
    
    def train_lightgbm(self):
        """Train LightGBM model."""
        print("\n[3/4] Training LightGBM...")
        
        lgbm_model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            random_state=42,
            verbose=-1
        )
        
        lgbm_model.fit(self.X_train, self.y_train)
        y_pred = lgbm_model.predict(self.X_test)
        
        self.models['LightGBM'] = lgbm_model
        self.predictions['LightGBM'] = y_pred
        
        self._evaluate_model('LightGBM', y_pred)
        
        return lgbm_model
    
    def train_ensemble(self):
        """Train Ensemble (Voting) model."""
        print("\n[4/4] Training Ensemble (Voting)...")
        
        ensemble = VotingClassifier(
            estimators=[
                ('rf', self.models['Random Forest']),
                ('xgb', self.models['XGBoost']),
                ('lgbm', self.models['LightGBM'])
            ],
            voting='hard'
        )
        
        ensemble.fit(self.X_train, self.y_train)
        y_pred = ensemble.predict(self.X_test)
        
        self.models['Ensemble'] = ensemble
        self.predictions['Ensemble'] = y_pred
        
        self._evaluate_model('Ensemble', y_pred)
        
        return ensemble
    
    def _evaluate_model(self, model_name: str, y_pred):
        """Evaluate model performance."""
        print(f"\n=== {model_name} Results ===")
        print(classification_report(self.y_test, y_pred))
        
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix(self.y_test, y_pred)}")
        
        self.performance[model_name] = {
            'accuracy': accuracy,
            'f1_score': f1
        }
    
    def save_models(self, output_dir: str = "models"):
        """Save trained models to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = f"{output_dir}/{model_name.lower().replace(' ', '_')}.pkl"
            joblib.dump(model, filename)
            print(f"✅ Saved {model_name} to {filename}")
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get performance summary as DataFrame."""
        summary = pd.DataFrame({
            'Model': list(self.performance.keys()),
            'Accuracy': [perf['accuracy'] for perf in self.performance.values()],
            'F1-Score': [perf['f1_score'] for perf in self.performance.values()]
        })
        return summary
    
    def run_pipeline(self) -> dict:
        """Run the complete model training pipeline."""
        print("\n" + "="*60)
        print("PHASE 2: MODEL TRAINING")
        print("="*60 + "\n")
        
        self.load_data()
        X, y, feature_cols = self.prepare_features()
        self.split_data(X, y)
        
        # Train all models
        self.train_random_forest()
        self.train_xgboost()
        self.train_lightgbm()
        self.train_ensemble()
        
        # Save models
        self.save_models()
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        summary = self.get_performance_summary()
        print(summary.to_string(index=False))
        
        print("\n✅ Model training completed successfully!")
        
        return {
            'models': self.models,
            'predictions': self.predictions,
            'performance': self.performance,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'feature_names': feature_cols
        }


def main():
    """Main execution function."""
    # Path to feature data
    feature_data_path = "data/processed/feature_dataset.csv"
    
    # Initialize and run trainer
    trainer = ModelTrainer(feature_data_path)
    results = trainer.run_pipeline()
    
    print(f"\n✅ All models trained and saved!")
    print(f"Best model: {max(results['performance'].items(), key=lambda x: x[1]['accuracy'])[0]}")


if __name__ == "__main__":
    main()