"""
Advanced SHAP-Based Predictions Module
Phase 4: Multi-target predictions with SHAP explainability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os
import warnings
warnings.filterwarnings('ignore')


class AdvancedPredictionEngine:
    """
    Advanced prediction engine with SHAP explainability for multiple health targets.
    Includes Sleep Quality, CV Risk, Activity, Stress, and Recovery predictions.
    """
    
    def __init__(self, data_path: str, outputs_dir: str = "outputs/advanced_predictions"):
        """
        Initialize the prediction engine.
        
        Args:
            data_path: Path to feature dataset
            outputs_dir: Directory to save outputs
        """
        self.data_path = data_path
        self.outputs_dir = outputs_dir
        self.df = None
        self.models = {}
        self.results = {}
        
        # Create output directory
        os.makedirs(outputs_dir, exist_ok=True)
    
    def load_and_prepare_data(self):
        """Load and prepare dataset for predictions."""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        
        # One-hot encode categorical features
        categorical_features = ['gender']
        self.df = pd.get_dummies(self.df, columns=categorical_features, drop_first=True)
        
        print(f"Dataset shape: {self.df.shape}")
        return self.df
    
    def _get_features(self, exclude_cols: list) -> list:
        """Get feature columns excluding specified columns."""
        base_exclude = ['user_id', 'day', 'Anomaly', 'Risk_Level', 'Risk_Level_Cluster',
                       'Age (years)', 'Height (meter)', 'Weight (kg)', 'BMI', 'Health_Score']
        all_exclude = list(set(base_exclude + exclude_cols))
        return [col for col in self.df.columns if col not in all_exclude]
    
    def _train_and_evaluate(self, X_train, X_test, y_train, y_test, 
                           model_type: str, model_name: str):
        """Train model and compute metrics."""
        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=200, max_depth=10, 
                                         random_state=42, n_jobs=-1)
        else:  # gb
            model = GradientBoostingRegressor(n_estimators=200, max_depth=8, 
                                             random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
        
        return model, y_pred, {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    def _create_shap_plots(self, model, X_test_sample, model_name: str, 
                          plot_type: str = "bar"):
        """Create and save SHAP plots."""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_sample)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_sample, plot_type=plot_type, show=False)
        plt.title(f"{model_name} - SHAP Feature Impact", fontweight='bold', fontsize=14)
        plt.tight_layout()
        
        filename = f"shap_{model_name.lower().replace(' ', '_').replace('-', '_')}.png"
        plt.savefig(os.path.join(self.outputs_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        return shap_values
    
    def predict_sleep_quality(self):
        """Predict sleep quality score."""
        print("\n" + "="*70)
        print("1. SLEEP QUALITY PREDICTION")
        print("="*70)
        
        # Create target
        self.df['Sleep_Quality'] = (
            (self.df['sleep_min'] / 600) * 40 +
            (self.df['spO2'] / 100) * 30 +
            (1 - self.df['heart_rate'] / 120) * 20 +
            (self.df['steps'] / 10000) * 10
        ) * 100
        
        # Prepare features
        exclude = ['Sleep_Quality', 'sleep_min']
        features = self._get_features(exclude)
        
        X = self.df[features].fillna(0)
        y = self.df['Sleep_Quality']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model, y_pred, metrics = self._train_and_evaluate(
            X_train, X_test, y_train, y_test, 'rf', 'Sleep Quality'
        )
        
        # SHAP analysis
        X_sample = X_test.iloc[:500]
        shap_values = self._create_shap_plots(model, X_sample, 'Sleep Quality')
        
        # Store results
        self.models['sleep_quality'] = model
        self.results['sleep_quality'] = {
            'metrics': metrics,
            'predictions': y_pred,
            'actual': y_test
        }
        
        return model, metrics
    
    def predict_cv_risk(self):
        """Predict cardiovascular risk score."""
        print("\n" + "="*70)
        print("2. CARDIOVASCULAR RISK PREDICTION")
        print("="*70)
        
        # Create target
        self.df['CV_Risk_Score'] = (
            (self.df['heart_rate'] / 120) * 30 +
            (self.df['systolic_bp'] / 140) * 25 +
            (self.df['diastolic_bp'] / 90) * 20 +
            (1 - self.df['spO2'] / 100) * 15 +
            (1 - self.df['steps'] / 10000) * 10
        ) * 100
        
        # Prepare features
        exclude = ['CV_Risk_Score', 'heart_rate', 'systolic_bp', 'diastolic_bp']
        features = self._get_features(exclude)
        
        X = self.df[features].fillna(0)
        y = self.df['CV_Risk_Score']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model, y_pred, metrics = self._train_and_evaluate(
            X_train, X_test, y_train, y_test, 'gb', 'CV Risk'
        )
        
        # SHAP analysis
        X_sample = X_test.iloc[:500]
        shap_values = self._create_shap_plots(model, X_sample, 'CV Risk', plot_type='dot')
        
        # Store results
        self.models['cv_risk'] = model
        self.results['cv_risk'] = {
            'metrics': metrics,
            'predictions': y_pred,
            'actual': y_test
        }
        
        return model, metrics
    
    def predict_next_day_activity(self):
        """Predict next-day step count."""
        print("\n" + "="*70)
        print("3. NEXT-DAY ACTIVITY PREDICTION")
        print("="*70)
        
        # Create next-day target
        df_sorted = self.df.sort_values(['user_id', 'day']).reset_index(drop=True)
        df_sorted['next_day_steps'] = df_sorted.groupby('user_id')['steps'].shift(-1)
        df_activity = df_sorted.dropna(subset=['next_day_steps'])
        
        # Prepare features
        exclude = ['next_day_steps', 'steps', 'distance_km']
        features = self._get_features(exclude)
        
        X = df_activity[features].fillna(0)
        y = df_activity['next_day_steps']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model, y_pred, metrics = self._train_and_evaluate(
            X_train, X_test, y_train, y_test, 'rf', 'Next-Day Activity'
        )
        
        print(f"Next-Day Activity - MAE: {metrics['mae']:.0f} steps")
        
        # SHAP analysis
        X_sample = X_test.iloc[:500]
        shap_values = self._create_shap_plots(model, X_sample, 'Next-Day Activity')
        
        # Store results
        self.models['next_day_activity'] = model
        self.results['next_day_activity'] = {
            'metrics': metrics,
            'predictions': y_pred,
            'actual': y_test
        }
        
        return model, metrics
    
    def predict_stress_level(self):
        """Predict stress index."""
        print("\n" + "="*70)
        print("4. STRESS LEVEL PREDICTION")
        print("="*70)
        
        # Create target
        self.df['Stress_Index'] = (
            (self.df['heart_rate'] / 120) * 25 +
            (self.df['screen_min'] / 480) * 20 +
            (1 - self.df['sleep_min'] / 600) * 25 +
            (self.df['heart_rate_roll_std'] / 20) * 15 +
            (1 - self.df['spO2'] / 100) * 15
        ) * 100
        
        # Prepare features
        exclude = ['Stress_Index']
        features = self._get_features(exclude)
        
        X = self.df[features].fillna(0)
        y = self.df['Stress_Index']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model, y_pred, metrics = self._train_and_evaluate(
            X_train, X_test, y_train, y_test, 'gb', 'Stress Level'
        )
        
        # SHAP analysis with waterfall plot for individual
        X_sample = X_test.iloc[:500]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title("Stress Level - SHAP Feature Impact", fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.outputs_dir, "shap_stress_level.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Waterfall plot for first prediction
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_sample.iloc[0],
                feature_names=X_sample.columns.tolist()
            ),
            show=False
        )
        plt.title("Individual Stress Prediction Explanation", fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.outputs_dir, "shap_stress_waterfall.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store results
        self.models['stress'] = model
        self.results['stress'] = {
            'metrics': metrics,
            'predictions': y_pred,
            'actual': y_test
        }
        
        return model, metrics
    
    def predict_recovery_time(self):
        """Predict recovery time."""
        print("\n" + "="*70)
        print("5. RECOVERY TIME PREDICTION")
        print("="*70)
        
        # Create recovery features
        df_rec = self.df.copy()
        df_rec['needs_recovery'] = (
            (df_rec['heart_rate'] > 90) |
            (df_rec['spO2'] < 95) |
            (df_rec['sleep_min'] < 360)
        ).astype(int)
        
        df_rec = df_rec.sort_values(['user_id', 'day']).reset_index(drop=True)
        df_rec['recovery_days'] = df_rec.groupby('user_id')['needs_recovery'].rolling(5).sum().reset_index(0, drop=True)
        df_rec = df_rec.dropna(subset=['recovery_days'])
        
        # Prepare features
        exclude = ['needs_recovery', 'recovery_days']
        features = self._get_features(exclude)
        
        X = df_rec[features].fillna(0)
        y = df_rec['recovery_days']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model, y_pred, metrics = self._train_and_evaluate(
            X_train, X_test, y_train, y_test, 'rf', 'Recovery Time'
        )
        
        print(f"Recovery Time - MAE: {metrics['mae']:.2f} days")
        
        # SHAP analysis
        X_sample = X_test.iloc[:500]
        shap_values = self._create_shap_plots(model, X_sample, 'Recovery Time')
        
        # Store results
        self.models['recovery'] = model
        self.results['recovery'] = {
            'metrics': metrics,
            'predictions': y_pred,
            'actual': y_test
        }
        
        return model, metrics
    
    def generate_personalized_recommendations(self, user_index: int = 0, top_n: int = 10):
        """Generate personalized health recommendations using SHAP."""
        print("\n" + "="*70)
        print("6. PERSONALIZED RECOMMENDATIONS")
        print("="*70)
        
        if 'stress' not in self.models:
            print("⚠️  Run stress prediction first to generate recommendations")
            return None
        
        # Get stress model and data
        model = self.models['stress']
        X_test = self.results['stress']['actual'].index
        
        # Get SHAP values for a sample
        exclude = ['Stress_Index']
        features = self._get_features(exclude)
        X = self.df[features].fillna(0)
        
        # Get sample
        X_sample = X.iloc[user_index:user_index+1]
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Generate recommendations
        impacts = np.abs(shap_values[0])
        top_indices = np.argsort(impacts)[-top_n:][::-1]
        
        recommendations = []
        for idx in top_indices:
            feature = X_sample.columns[idx]
            shap_val = shap_values[0][idx]
            feat_val = X_sample.iloc[0, idx]
            
            if shap_val > 0:
                action = "Reduce" if "negative" not in feature.lower() else "Improve"
            else:
                action = "Maintain"
            
            priority = 'High' if abs(shap_val) > np.percentile(impacts, 75) else 'Medium'
            
            recommendations.append({
                'Feature': feature,
                'Impact': abs(shap_val),
                'Current_Value': feat_val,
                'Recommendation': f"{action} {feature}",
                'Priority': priority
            })
        
        rec_df = pd.DataFrame(recommendations)
        print("\n📋 PERSONALIZED HEALTH RECOMMENDATIONS:")
        print(rec_df.to_string(index=False))
        
        # Save recommendations
        rec_path = os.path.join(self.outputs_dir, "personalized_recommendations.csv")
        rec_df.to_csv(rec_path, index=False)
        print(f"\n✅ Recommendations saved to {rec_path}")
        
        return rec_df
    
    def create_summary_visualization(self):
        """Create summary visualization of all predictions."""
        print("\n" + "="*70)
        print("7. CREATING SUMMARY VISUALIZATION")
        print("="*70)
        
        # Prepare summary data
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                'Prediction Task': name.replace('_', ' ').title(),
                'R² Score': result['metrics']['r2'],
                'MAE': result['metrics']['mae']
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² Score
        summary_df.plot(x='Prediction Task', y='R² Score', kind='barh', 
                       ax=axes[0], legend=False, color='steelblue')
        axes[0].set_title('Model Performance - R² Score', fontweight='bold', fontsize=12)
        axes[0].set_xlabel('R² Score')
        axes[0].grid(axis='x', alpha=0.3)
        
        # MAE
        summary_df.plot(x='Prediction Task', y='MAE', kind='barh', 
                       ax=axes[1], legend=False, color='coral')
        axes[1].set_title('Model Performance - Mean Absolute Error', fontweight='bold', fontsize=12)
        axes[1].set_xlabel('MAE')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        summary_path = os.path.join(self.outputs_dir, "predictions_summary.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Summary visualization saved to {summary_path}")
        
        return summary_df
    
    def run_complete_pipeline(self):
        """Run all predictions in sequence."""
        print("\n" + "🚀"*35)
        print("ADVANCED SHAP-BASED PREDICTIONS PIPELINE")
        print("🚀"*35 + "\n")
        
        # Load data
        self.load_and_prepare_data()
        
        # Run all predictions
        self.predict_sleep_quality()
        self.predict_cv_risk()
        self.predict_next_day_activity()
        self.predict_stress_level()
        self.predict_recovery_time()
        
        # Generate recommendations
        self.generate_personalized_recommendations()
        
        # Create summary
        summary_df = self.create_summary_visualization()
        
        print("\n" + "🎉"*35)
        print("ALL PREDICTIONS COMPLETED SUCCESSFULLY!")
        print("🎉"*35)
        
        print("\n✅ Generated files:")
        print(f"   - {self.outputs_dir}/shap_sleep_quality.png")
        print(f"   - {self.outputs_dir}/shap_cv_risk.png")
        print(f"   - {self.outputs_dir}/shap_next_day_activity.png")
        print(f"   - {self.outputs_dir}/shap_stress_level.png")
        print(f"   - {self.outputs_dir}/shap_stress_waterfall.png")
        print(f"   - {self.outputs_dir}/shap_recovery_time.png")
        print(f"   - {self.outputs_dir}/personalized_recommendations.csv")
        print(f"   - {self.outputs_dir}/predictions_summary.png")
        
        return self.models, self.results, summary_df


def main():
    """Main execution function."""
    # Path to feature dataset
    data_path = "data/processed/feature_dataset.csv"
    
    # Initialize prediction engine
    engine = AdvancedPredictionEngine(data_path)
    
    # Run complete pipeline
    models, results, summary = engine.run_complete_pipeline()
    
    print("\n✨ Advanced predictions ready for analysis!")


if __name__ == "__main__":
    main()