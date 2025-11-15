"""
Deep Learning SHAP Explainability Module
Phase 4: SHAP-based explanations for TensorFlow/Keras models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import warnings
from typing import Dict, List, Tuple, Optional, Union
warnings.filterwarnings('ignore')

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available. DNN SHAP will be limited.")


class DNNSHAPExplainer:
    """
    SHAP explainability for deep neural network models.
    Handles multiple architectures: standard DNNs, transformers, CNNs, ResNets.
    """
    
    def __init__(self, outputs_dir: str = "outputs/dnn_shap"):
        """
        Initialize the DNN SHAP explainer.
        
        Args:
            outputs_dir: Directory to save SHAP visualizations
        """
        self.outputs_dir = outputs_dir
        self.explainers = {}
        self.shap_values_cache = {}
        
        # Create output directory
        os.makedirs(outputs_dir, exist_ok=True)
        
        # Check TensorFlow availability
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for DNN SHAP explainability")
    
    def _prepare_data(self, X, max_samples: int = 200) -> np.ndarray:
        """Convert data to numpy and sample if needed."""
        # Convert to numpy if pandas
        if hasattr(X, 'values'):
            X = X.values
        else:
            X = np.asarray(X)
        
        # Sample if too large
        if X.shape[0] > max_samples:
            indices = np.random.choice(X.shape[0], max_samples, replace=False)
            X = X[indices]
        
        return X
    
    def _create_explainer(self, model, background_data, 
                         explainer_type: str = 'auto') -> shap.Explainer:
        """
        Create appropriate SHAP explainer based on model type.
        
        Args:
            model: Trained Keras model
            background_data: Background data for explainer
            explainer_type: 'auto', 'deep', 'gradient', or 'kernel'
        """
        if explainer_type == 'auto':
            # Try DeepExplainer first (fastest for deep models)
            try:
                return shap.DeepExplainer(model, background_data)
            except Exception as e:
                print(f"DeepExplainer failed ({e}), trying GradientExplainer...")
                try:
                    return shap.GradientExplainer(model, background_data)
                except Exception as e2:
                    print(f"GradientExplainer failed ({e2}), using KernelExplainer...")
                    return shap.KernelExplainer(model.predict, background_data)
        
        elif explainer_type == 'deep':
            return shap.DeepExplainer(model, background_data)
        elif explainer_type == 'gradient':
            return shap.GradientExplainer(model, background_data)
        elif explainer_type == 'kernel':
            return shap.KernelExplainer(model.predict, background_data)
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
    
    def explain_standard_dnn(self, model, X_train, X_test, 
                            model_name: str = "Standard DNN",
                            background_size: int = 200,
                            test_size: int = 200):
        """
        Explain a standard feedforward DNN.
        
        Args:
            model: Trained Keras model
            X_train: Training data for background
            X_test: Test data for explanation
            model_name: Name for plots
            background_size: Number of background samples
            test_size: Number of test samples to explain
        """
        print(f"\n{'='*70}")
        print(f"🔹 SHAP EXPLAINABILITY FOR: {model_name}")
        print(f"{'='*70}")
        
        try:
            # Prepare data
            X_train_np = self._prepare_data(X_train, background_size)
            X_test_np = self._prepare_data(X_test, test_size)
            
            print(f"Background shape: {X_train_np.shape}")
            print(f"Test shape: {X_test_np.shape}")
            
            # Create explainer
            explainer = self._create_explainer(model, X_train_np)
            
            # Compute SHAP values
            print("Computing SHAP values...")
            shap_values = explainer.shap_values(X_test_np)
            
            # Handle multi-class case (list of arrays)
            if isinstance(shap_values, list):
                print(f"Multi-class detected: {len(shap_values)} classes")
                # Use class 1 (high risk) or average all classes
                shap_values_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_values_plot = shap_values
            
            # Store results
            self.shap_values_cache[model_name] = {
                'shap_values': shap_values,
                'test_data': X_test_np
            }
            
            # Create visualizations
            self._create_shap_plots(shap_values_plot, X_test_np, model_name)
            
            print(f"✅ {model_name} SHAP analysis completed!")
            return shap_values
            
        except Exception as e:
            print(f"❌ SHAP failed for {model_name}: {e}")
            return None
    
    def explain_cnn_model(self, model, X_train, X_test,
                         model_name: str = "CNN Model",
                         input_shape: Tuple = None,
                         background_size: int = 100,
                         test_size: int = 200):
        """
        Explain CNN models with special handling for 2D/3D inputs.
        
        Args:
            model: Trained Keras CNN model
            X_train: Training data
            X_test: Test data
            model_name: Name for plots
            input_shape: Expected input shape (H, W, C)
            background_size: Number of background samples
            test_size: Number of test samples
        """
        print(f"\n{'='*70}")
        print(f"🔹 SHAP EXPLAINABILITY FOR: {model_name}")
        print(f"{'='*70}")
        
        try:
            # Get input shape from model if not provided
            if input_shape is None:
                input_shape = model.input_shape[1:]  # Skip batch dimension
            
            print(f"Model expected input shape: {input_shape}")
            
            # Convert to numpy
            X_train_np = self._prepare_data(X_train, background_size)
            X_test_np = self._prepare_data(X_test, test_size)
            
            # Reshape to match model input
            H, W = input_shape[0], input_shape[1]
            C = input_shape[2] if len(input_shape) > 2 else 1
            
            X_train_reshaped = self._reshape_for_cnn(X_train_np, H, W, C)
            X_test_reshaped = self._reshape_for_cnn(X_test_np, H, W, C)
            
            print(f"Reshaped training data: {X_train_reshaped.shape}")
            print(f"Reshaped test data: {X_test_reshaped.shape}")
            
            # Create explainer
            explainer = self._create_explainer(model, X_train_reshaped)
            
            # Compute SHAP values
            print("Computing SHAP values...")
            shap_values = explainer.shap_values(X_test_reshaped)
            
            # Flatten for visualization
            if isinstance(shap_values, list):
                shap_flat = np.mean([s.reshape(s.shape[0], -1) for s in shap_values], axis=0)
            else:
                shap_flat = shap_values.reshape(shap_values.shape[0], -1)
            
            X_test_flat = X_test_reshaped.reshape(X_test_reshaped.shape[0], -1)
            
            # Ensure matching dimensions
            n = min(shap_flat.shape[1], X_test_flat.shape[1])
            shap_flat = shap_flat[:, :n]
            X_test_flat = X_test_flat[:, :n]
            
            # Create feature names
            feature_names = [f"pixel_{i}" for i in range(n)]
            
            # Store results
            self.shap_values_cache[model_name] = {
                'shap_values': shap_flat,
                'test_data': X_test_flat,
                'feature_names': feature_names
            }
            
            # Create visualizations
            self._create_shap_plots(shap_flat, X_test_flat, model_name, 
                                   feature_names=feature_names)
            
            print(f"✅ {model_name} SHAP analysis completed!")
            return shap_values
            
        except Exception as e:
            print(f"❌ SHAP failed for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _reshape_for_cnn(self, X, H, W, C):
        """Reshape data to CNN input format (samples, H, W, C)."""
        X = X.reshape(X.shape[0], -1)
        needed = H * W * C
        
        # Pad or truncate to match required size
        if X.shape[1] < needed:
            X = np.pad(X, ((0, 0), (0, needed - X.shape[1])), mode='constant')
        elif X.shape[1] > needed:
            X = X[:, :needed]
        
        return X.reshape(X.shape[0], H, W, C)
    
    def _create_shap_plots(self, shap_values, X_test, model_name,
                          feature_names=None, max_display=15):
        """Create and save SHAP visualizations."""
        
        # Bar plot - Feature importance
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, 
                         feature_names=feature_names,
                         plot_type="bar", 
                         show=False, 
                         max_display=max_display)
        plt.title(f"{model_name} - SHAP Feature Importance", 
                 fontweight='bold', fontsize=14)
        plt.tight_layout()
        
        filename = f"shap_{model_name.lower().replace(' ', '_').replace('-', '_')}_bar.png"
        plt.savefig(os.path.join(self.outputs_dir, filename), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Beeswarm plot - Detailed impact
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test,
                         feature_names=feature_names,
                         show=False,
                         max_display=max_display)
        plt.title(f"{model_name} - SHAP Summary Plot", 
                 fontweight='bold', fontsize=14)
        plt.tight_layout()
        
        filename = f"shap_{model_name.lower().replace(' ', '_').replace('-', '_')}_summary.png"
        plt.savefig(os.path.join(self.outputs_dir, filename), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 SHAP plots saved for {model_name}")
    
    def create_model_comparison(self, model_performances: Dict[str, Dict]):
        """
        Create comparison visualization across all models.
        
        Args:
            model_performances: Dict mapping model names to performance metrics
                              e.g., {'Model1': {'accuracy': 0.95, 'loss': 0.12}}
        """
        print(f"\n{'='*70}")
        print("📊 CREATING MODEL COMPARISON VISUALIZATION")
        print(f"{'='*70}")
        
        df = pd.DataFrame(model_performances).T
        
        fig, axes = plt.subplots(1, len(df.columns), figsize=(5*len(df.columns), 6))
        
        if len(df.columns) == 1:
            axes = [axes]
        
        for idx, col in enumerate(df.columns):
            df[col].plot(kind='barh', ax=axes[idx], color='steelblue')
            axes[idx].set_title(f'{col.title()}', fontweight='bold', fontsize=12)
            axes[idx].set_xlabel(col.title())
            axes[idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.outputs_dir, 'model_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Model comparison saved!")
    
    def explain_all_models(self, models_dict: Dict, X_train, X_test,
                          model_configs: Dict = None):
        """
        Explain multiple models at once.
        
        Args:
            models_dict: Dict mapping model names to model objects
            X_train: Training data
            X_test: Test data
            model_configs: Optional dict with specific configs per model
                          e.g., {'CNN': {'input_shape': (6, 6, 1)}}
        """
        print(f"\n{'🚀'*35}")
        print("COMPREHENSIVE DNN SHAP ANALYSIS")
        print(f"{'🚀'*35}\n")
        
        results = {}
        
        for model_name, model in models_dict.items():
            print(f"\nProcessing: {model_name}")
            
            # Get model-specific config
            config = model_configs.get(model_name, {}) if model_configs else {}
            
            # Determine model type
            if 'cnn' in model_name.lower() or 'vgg' in model_name.lower():
                shap_vals = self.explain_cnn_model(
                    model, X_train, X_test, 
                    model_name=model_name,
                    **config
                )
            else:
                shap_vals = self.explain_standard_dnn(
                    model, X_train, X_test,
                    model_name=model_name
                )
            
            results[model_name] = shap_vals
        
        print(f"\n{'🎉'*35}")
        print("SHAP ANALYSIS COMPLETED FOR ALL MODELS!")
        print(f"{'🎉'*35}")
        
        return results
    
    def get_top_features(self, model_name: str, top_n: int = 10) -> pd.DataFrame:
        """
        Get top N most important features for a model.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to return
        """
        if model_name not in self.shap_values_cache:
            print(f"⚠️  No SHAP values found for {model_name}")
            return None
        
        data = self.shap_values_cache[model_name]
        shap_values = data['shap_values']
        
        # Calculate mean absolute SHAP value per feature
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Get feature names
        feature_names = data.get('feature_names', 
                                [f"feature_{i}" for i in range(len(mean_shap))])
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': mean_shap
        }).sort_values('Importance', ascending=False).head(top_n)
        
        return importance_df
    
    def create_feature_importance_report(self, output_file: str = None):
        """Create comprehensive feature importance report for all models."""
        
        if not self.shap_values_cache:
            print("⚠️  No SHAP values computed yet")
            return
        
        report = []
        
        for model_name in self.shap_values_cache.keys():
            top_features = self.get_top_features(model_name, top_n=10)
            if top_features is not None:
                report.append(f"\n{'='*70}")
                report.append(f"TOP FEATURES FOR: {model_name}")
                report.append(f"{'='*70}\n")
                report.append(top_features.to_string(index=False))
        
        report_text = '\n'.join(report)
        
        # Print to console
        print(report_text)
        
        # Save to file
        if output_file is None:
            output_file = os.path.join(self.outputs_dir, 'feature_importance_report.txt')
        
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"\n✅ Feature importance report saved to {output_file}")


def main():
    """Example usage of DNNSHAPExplainer."""
    
    print("DNN SHAP Explainer - Example Usage")
    print("="*70)
    
    # This is a template - replace with actual model loading
    print("""
To use this module:

1. Train your DNN models
2. Load them and prepare data
3. Initialize explainer:
   
   explainer = DNNSHAPExplainer('outputs/dnn_shap')

4. Explain models:
   
   # For standard DNN
   explainer.explain_standard_dnn(model, X_train, X_test, 'My DNN')
   
   # For CNN
   explainer.explain_cnn_model(model, X_train, X_test, 'My CNN',
                               input_shape=(6, 6, 1))
   
   # Or explain all at once
   models = {
       'DNN': dnn_model,
       'CNN': cnn_model,
       'ResNet': resnet_model
   }
   explainer.explain_all_models(models, X_train, X_test)

5. Generate reports:
   
   explainer.create_feature_importance_report()
    """)


if __name__ == "__main__":
    main()