"""
SHAP Explainability Module
Phase 3: Generate SHAP-based explanations for trained models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """Generates SHAP explanations for trained ML models."""
    
    def __init__(self, models_dir: str = "models", outputs_dir: str = "outputs"):
        """
        Initialize the SHAP explainer.
        
        Args:
            models_dir: Directory containing trained models
            outputs_dir: Directory to save SHAP visualizations
        """
        self.models_dir = models_dir
        self.outputs_dir = outputs_dir
        self.models = {}
        self.X_test = None
        self.y_test = None
        
        # Create outputs directory
        os.makedirs(outputs_dir, exist_ok=True)
    
    def load_models(self):
        """Load all trained models."""
        print("Loading trained models...")
        
        model_files = {
            'Random Forest': 'random_forest.pkl',
            'XGBoost': 'xgboost.pkl',
            'LightGBM': 'lightgbm.pkl',
            'Ensemble': 'ensemble.pkl'
        }
        
        for name, filename in model_files.items():
            filepath = os.path.join(self.models_dir, filename)
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
                print(f"✅ Loaded {name}")
            else:
                print(f"⚠️  {name} not found at {filepath}")
        
        return self.models
    
    def load_test_data(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Load test data for SHAP analysis."""
        self.X_test = X_test
        self.y_test = y_test
        print(f"✅ Test data loaded: {X_test.shape}")
    
    def explain_random_forest(self, sample_size: int = 500):
        """Generate SHAP explanations for Random Forest."""
        print("\n[1/4] Generating SHAP for Random Forest...")
        
        if 'Random Forest' not in self.models:
            print("⚠️  Random Forest model not loaded. Skipping...")
            return
        
        model = self.models['Random Forest']
        X_sample = self.X_test.sample(n=min(sample_size, len(self.X_test)), random_state=42)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Feature importance bar plot
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_sample, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title("Random Forest - Feature Importance (SHAP)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.outputs_dir}/shap_rf_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Detailed summary plot
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_sample, show=False)
        else:
            shap.summary_plot(shap_values, X_sample, show=False)
        plt.title("Random Forest - SHAP Summary Plot", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.outputs_dir}/shap_rf_detailed.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Random Forest SHAP plots saved")
        return shap_values
    
    def explain_xgboost(self, sample_size: int = 500):
        """Generate SHAP explanations for XGBoost."""
        print("\n[2/4] Generating SHAP for XGBoost...")
        
        if 'XGBoost' not in self.models:
            print("⚠️  XGBoost model not loaded. Skipping...")
            return
        
        model = self.models['XGBoost']
        X_sample = self.X_test.sample(n=min(sample_size, len(self.X_test)), random_state=42)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Feature importance bar plot
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_sample, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title("XGBoost - Feature Importance (SHAP)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.outputs_dir}/shap_xgb_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Detailed summary plot
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_sample, show=False)
        else:
            shap.summary_plot(shap_values, X_sample, show=False)
        plt.title("XGBoost - SHAP Summary Plot", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.outputs_dir}/shap_xgb_detailed.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ XGBoost SHAP plots saved")
        return shap_values
    
    def explain_lightgbm(self, sample_size: int = 500):
        """Generate SHAP explanations for LightGBM."""
        print("\n[3/4] Generating SHAP for LightGBM...")
        
        if 'LightGBM' not in self.models:
            print("⚠️  LightGBM model not loaded. Skipping...")
            return
        
        model = self.models['LightGBM']
        X_sample = self.X_test.sample(n=min(sample_size, len(self.X_test)), random_state=42)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Feature importance bar plot
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_sample, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title("LightGBM - Feature Importance (SHAP)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.outputs_dir}/shap_lgbm_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Detailed summary plot
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_sample, show=False)
        else:
            shap.summary_plot(shap_values, X_sample, show=False)
        plt.title("LightGBM - SHAP Summary Plot", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.outputs_dir}/shap_lgbm_detailed.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ LightGBM SHAP plots saved")
        return shap_values
    
    def explain_ensemble(self, sample_size: int = 500):
        """Generate SHAP explanations for Ensemble (averaged from base models)."""
        print("\n[4/4] Generating SHAP for Ensemble...")
        
        # Get SHAP values from all base models
        X_sample = self.X_test.sample(n=min(sample_size, len(self.X_test)), random_state=42)
        
        shap_values_list = []
        for model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
            if model_name in self.models:
                explainer = shap.TreeExplainer(self.models[model_name])
                shap_vals = explainer.shap_values(X_sample)
                shap_values_list.append(shap_vals)
        
        if not shap_values_list:
            print("⚠️  No base models available for ensemble SHAP. Skipping...")
            return
        
        # Average SHAP values
        if isinstance(shap_values_list[0], list):
            # Multi-class case
            avg_shap = []
            for i in range(len(shap_values_list[0])):
                class_shap = np.mean([sv[i] for sv in shap_values_list], axis=0)
                avg_shap.append(class_shap)
        else:
            # Binary or single output case
            avg_shap = np.mean(shap_values_list, axis=0)
        
        # Feature importance bar plot
        plt.figure(figsize=(12, 8))
        if isinstance(avg_shap, list):
            shap.summary_plot(avg_shap[1], X_sample, plot_type="bar", show=False)
        else:
            shap.summary_plot(avg_shap, X_sample, plot_type="bar", show=False)
        plt.title("Ensemble - Feature Importance (SHAP)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.outputs_dir}/shap_ensemble_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Detailed summary plot
        plt.figure(figsize=(12, 8))
        if isinstance(avg_shap, list):
            shap.summary_plot(avg_shap[1], X_sample, show=False)
        else:
            shap.summary_plot(avg_shap, X_sample, show=False)
        plt.title("Ensemble - SHAP Summary Plot", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.outputs_dir}/shap_ensemble_detailed.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Ensemble SHAP plots saved")
        return avg_shap
    
    def create_comparison_plot(self, performance_df: pd.DataFrame):
        """Create model comparison visualization."""
        print("\nCreating model comparison plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy comparison
        performance_df.plot(x='Model', y='Accuracy', kind='bar', 
                           ax=axes[0], legend=False, color='skyblue')
        axes[0].set_title('Model Accuracy Comparison', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
        
        # F1-Score comparison
        performance_df.plot(x='Model', y='F1-Score', kind='bar', 
                           ax=axes[1], legend=False, color='salmon')
        axes[1].set_title('Model F1-Score Comparison', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('Weighted F1-Score')
        axes[1].set_ylim([0, 1])
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{self.outputs_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Model comparison plot saved to {self.outputs_dir}/model_comparison.png")
    
    def run_pipeline(self, X_test: pd.DataFrame, y_test: pd.Series, 
                    performance_df: pd.DataFrame = None):
        """Run the complete SHAP explainability pipeline."""
        print("\n" + "="*60)
        print("PHASE 3: SHAP EXPLAINABILITY")
        print("="*60 + "\n")
        
        self.load_models()
        self.load_test_data(X_test, y_test)
        
        # Generate SHAP for all models
        self.explain_random_forest()
        self.explain_xgboost()
        self.explain_lightgbm()
        self.explain_ensemble()
        
        # Create comparison plot if performance data provided
        if performance_df is not None:
            self.create_comparison_plot(performance_df)
        
        print("\n" + "="*60)
        print("✅ SHAP explainability analysis completed!")
        print(f"✅ All plots saved to {self.outputs_dir}/")
        print("="*60)


def main():
    """Main execution function (requires trained models and test data)."""
    from model_training import ModelTrainer
    
    # Train models first
    print("Training models...")
    trainer = ModelTrainer("data/processed/feature_dataset.csv")
    results = trainer.run_pipeline()
    
    # Run SHAP analysis
    explainer = SHAPExplainer()
    performance_df = trainer.get_performance_summary()
    explainer.run_pipeline(
        X_test=results['X_test'],
        y_test=results['y_test'],
        performance_df=performance_df
    )


if __name__ == "__main__":
    main()