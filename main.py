"""
Main Pipeline Script
Runs the complete wearable data fusion pipeline from start to finish
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.shap_explainability import SHAPExplainer
from src.utils import create_directory, plot_correlation_heatmap


def run_complete_pipeline(raw_data_path: str = "data/raw/SmartHealthcare_Dataset.csv"):
    """
    Run the complete pipeline from raw data to SHAP explanations.
    
    Args:
        raw_data_path: Path to the raw dataset
    """
    
    print("\n" + "🚀"*30)
    print("WEARABLE HEALTH FUSION - COMPLETE PIPELINE")
    print("🚀"*30 + "\n")
    
    # Ensure all directories exist
    print("Setting up directories...")
    for directory in ['data/raw', 'data/cleaned', 'data/processed', 'outputs', 'models']:
        create_directory(directory)
    
    # ===== PHASE 0: Data Preprocessing =====
    print("\n" + "="*60)
    print("STARTING PHASE 0: DATA PREPROCESSING")
    print("="*60)
    
    preprocessor = DataPreprocessor(raw_data_path)
    df_cleaned = preprocessor.run_pipeline()
    
    # Create correlation heatmap
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) > 2:
        plot_correlation_heatmap(
            df_cleaned, 
            columns=numeric_cols[:15],  # Limit to first 15 numeric columns
            output_path="outputs/correlation_heatmap.png"
        )
    
    # ===== PHASE 1: Feature Engineering =====
    print("\n" + "="*60)
    print("STARTING PHASE 1: FEATURE ENGINEERING")
    print("="*60)
    
    engineer = FeatureEngineer("data/cleaned/canonical_dataset.csv")
    df_features = engineer.run_pipeline()
    
    # ===== PHASE 2: Model Training =====
    print("\n" + "="*60)
    print("STARTING PHASE 2: MODEL TRAINING")
    print("="*60)
    
    trainer = ModelTrainer("data/processed/feature_dataset.csv")
    results = trainer.run_pipeline()
    
    # ===== PHASE 3: SHAP Explainability =====
    print("\n" + "="*60)
    print("STARTING PHASE 3: SHAP EXPLAINABILITY")
    print("="*60)
    
    explainer = SHAPExplainer()
    performance_df = trainer.get_performance_summary()
    explainer.run_pipeline(
        X_test=results['X_test'],
        y_test=results['y_test'],
        performance_df=performance_df
    )
    
    # ===== Pipeline Complete =====
    print("\n" + "🎉"*30)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("🎉"*30)
    
    print("\n📊 Results Summary:")
    print("-" * 60)
    print(performance_df.to_string(index=False))
    print("-" * 60)
    
    print("\n📁 Output Files:")
    print("  ✅ Cleaned data: data/cleaned/canonical_dataset.csv")
    print("  ✅ Feature data: data/processed/feature_dataset.csv")
    print("  ✅ Models: models/")
    print("  ✅ Visualizations: outputs/")
    print("     - correlation_heatmap.png")
    print("     - shap_rf_summary.png & shap_rf_detailed.png")
    print("     - shap_xgb_summary.png & shap_xgb_detailed.png")
    print("     - shap_lgbm_summary.png & shap_lgbm_detailed.png")
    print("     - shap_ensemble_summary.png & shap_ensemble_detailed.png")
    print("     - model_comparison.png")
    
    print("\n✨ Project ready for GitHub!")


def run_phase(phase: str, **kwargs):
    """
    Run a specific phase of the pipeline.
    
    Args:
        phase: Phase name ('preprocessing', 'features', 'training', 'shap')
        **kwargs: Additional arguments for the phase
    """
    
    if phase == 'preprocessing':
        raw_data_path = kwargs.get('raw_data_path', 'data/raw/Smart Healthcare - Daily Lifestyle Dataset (Wearable device).csv')
        preprocessor = DataPreprocessor(raw_data_path)
        preprocessor.run_pipeline()
        
    elif phase == 'features':
        engineer = FeatureEngineer("data/cleaned/canonical_dataset.csv")
        engineer.run_pipeline()
        
    elif phase == 'training':
        trainer = ModelTrainer("data/processed/feature_dataset.csv")
        trainer.run_pipeline()
        
    elif phase == 'shap':
        # Need to load models and test data
        trainer = ModelTrainer("data/processed/feature_dataset_with_cluster.csv")
        trainer.load_data()
        X, y, _ = trainer.prepare_features()
        trainer.split_data(X, y)
        
        explainer = SHAPExplainer()
        explainer.run_pipeline(X_test=trainer.X_test, y_test=trainer.y_test)
        
    else:
        print(f"❌ Unknown phase: {phase}")
        print("Available phases: preprocessing, features, training, shap")


def main():
    """Main entry point with command-line arguments."""
    
    parser = argparse.ArgumentParser(
        description='Wearable Health Fusion Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py
  
  # Run specific phase
  python main.py --phase preprocessing
  python main.py --phase features
  python main.py --phase training
  python main.py --phase shap
  
  # Specify custom data path
  python main.py --data path/to/dataset.csv
        """
    )
    
    parser.add_argument(
        '--phase',
        type=str,
        choices=['all', 'preprocessing', 'features', 'training', 'shap'],
        default='all',
        help='Pipeline phase to run (default: all)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/raw/SmartHealthcare_Dataset.csv',
        help='Path to raw dataset (default: data/raw/SmartHealthcare_Dataset.csv)'
    )
    
    args = parser.parse_args()
    
    if args.phase == 'all':
        run_complete_pipeline(raw_data_path=args.data)
    else:
        run_phase(args.phase, raw_data_path=args.data)


if __name__ == "__main__":
    main()