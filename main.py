"""
Main Pipeline Script
Runs the complete wearable data fusion pipeline from start to finish
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import all modules
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from dnn_training import DNNTrainer
from cnn_training import CNNTrainer
from transformer_mlp_training import TransformerTrainer
from node_mlp_ensemble import NODETrainer
from resnet_training import ResNetTrainer
from shap_explainability import SHAPExplainer
from advanced_predictions import AdvancedPredictor
from model_comparison import create_comprehensive_comparison, print_model_summary
from utils import create_directory, plot_correlation_heatmap


def run_dnn_training():
    """Run DNN training phase."""
    print("\n" + "="*60)
    print("STARTING PHASE 2B: DNN TRAINING")
    print("="*60)
    
    trainer = ModelTrainer("data/processed/feature_dataset.csv")
    trainer.load_data()
    X, y, _ = trainer.prepare_features()
    trainer.split_data(X, y)
    
    dnn_trainer = DNNTrainer(
        X_train=trainer.X_train,
        X_test=trainer.X_test,
        y_train=trainer.y_train,
        y_test=trainer.y_test
    )
    
    dnn_trainer.run_pipeline()


def run_cnn_training():
    """Run CNN training phase."""
    print("\n" + "="*60)
    print("STARTING PHASE 2C: VGG-LIKE CNN TRAINING")
    print("="*60)
    
    trainer = ModelTrainer("data/processed/feature_dataset.csv")
    trainer.load_data()
    X, y, _ = trainer.prepare_features()
    trainer.split_data(X, y)
    
    cnn_trainer = CNNTrainer(
        X_train=trainer.X_train,
        X_test=trainer.X_test,
        y_train=trainer.y_train,
        y_test=trainer.y_test
    )
    
    cnn_trainer.run_pipeline()


def run_transformer_training():
    """Run Transformer training phase."""
    print("\n" + "="*60)
    print("STARTING PHASE 2D: TRANSFORMER-LIKE MLP TRAINING")
    print("="*60)
    
    trainer = ModelTrainer("data/processed/feature_dataset.csv")
    trainer.load_data()
    X, y, _ = trainer.prepare_features()
    trainer.split_data(X, y)
    
    transformer_trainer = TransformerTrainer(
        X_train=trainer.X_train,
        X_test=trainer.X_test,
        y_train=trainer.y_train,
        y_test=trainer.y_test
    )
    
    transformer_trainer.run_pipeline()


def run_node_training():
    """Run NODE training phase."""
    print("\n" + "="*60)
    print("STARTING PHASE 2E: NODE-LIKE ENSEMBLE TRAINING")
    print("="*60)
    
    trainer = ModelTrainer("data/processed/feature_dataset.csv")
    trainer.load_data()
    X, y, _ = trainer.prepare_features()
    trainer.split_data(X, y)
    
    node_trainer = NODETrainer(
        X_train=trainer.X_train,
        X_test=trainer.X_test,
        y_train=trainer.y_train,
        y_test=trainer.y_test
    )
    
    node_trainer.run_pipeline()


def run_resnet_training():
    """Run ResNet training phase."""
    print("\n" + "="*60)
    print("STARTING PHASE 2F: RESNET-STYLE MLP TRAINING")
    print("="*60)
    
    trainer = ModelTrainer("data/processed/feature_dataset.csv")
    trainer.load_data()
    X, y, _ = trainer.prepare_features()
    trainer.split_data(X, y)
    
    resnet_trainer = ResNetTrainer(
        X_train=trainer.X_train,
        X_test=trainer.X_test,
        y_train=trainer.y_train,
        y_test=trainer.y_test
    )
    
    resnet_trainer.run_pipeline()


def run_advanced_predictions():
    """Run advanced SHAP-based predictions."""
    print("\n" + "="*60)
    print("STARTING PHASE 5: ADVANCED PREDICTIONS")
    print("="*60)
    
    predictor = AdvancedPredictor()
    predictor.run_all_predictions()


def run_complete_pipeline(raw_data_path: str = "data/raw/SmartHealthcare_Dataset.csv"):
    """Run the complete pipeline from raw data to advanced predictions."""
    
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
            columns=numeric_cols[:15],
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
    
    # 2A: Traditional ML Models
    trainer = ModelTrainer("data/processed/feature_dataset.csv")
    results = trainer.run_pipeline()
    
    # 2B-F: Deep Learning Models
    run_dnn_training()
    run_cnn_training()
    run_transformer_training()
    run_node_training()
    run_resnet_training()
    
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
    
    # ===== PHASE 4: Model Comparison =====
    print("\n" + "="*60)
    print("STARTING PHASE 4: COMPREHENSIVE MODEL COMPARISON")
    print("="*60)
    
    create_comprehensive_comparison()
    print_model_summary()
    
    # ===== PHASE 5: Advanced Predictions =====
    print("\n" + "="*60)
    print("STARTING PHASE 5: ADVANCED PREDICTIONS")
    print("="*60)
    
    run_advanced_predictions()
    
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
    print("     - Traditional ML: random_forest.pkl, xgboost.pkl, lightgbm.pkl, ensemble.pkl")
    print("     - Deep Learning: dnn_model.h5, cnn_model.h5, transformer_model.h5")
    print("                     node_model.h5, resnet_model.h5")
    print("  ✅ Visualizations: outputs/")
    print("     - correlation_heatmap.png")
    print("     - Training histories: dnn_training_history.png, cnn_training_history.png,")
    print("                          transformer_training_history.png, node_training_history.png,")
    print("                          resnet_training_history.png")
    print("     - SHAP plots: shap_rf_summary.png, shap_xgb_summary.png, shap_lgbm_summary.png,")
    print("                   shap_ensemble_summary.png (summary & detailed)")
    print("     - Model comparison: model_comparison.png, all_models_comparison.png")
    print("     - Advanced predictions: shap_sleep_quality.png, shap_cv_risk.png,")
    print("                            shap_stress_level.png, shap_stress_waterfall.png")
    print("                            advanced_predictions_summary.png")
    
    print("\n✨ Project ready for GitHub!")


def run_phase(phase: str, **kwargs):
    """
    Run a specific phase of the pipeline.
    
    Args:
        phase: Phase name
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
    
    elif phase == 'dnn':
        run_dnn_training()
    
    elif phase == 'cnn':
        run_cnn_training()
    
    elif phase == 'transformer':
        run_transformer_training()
    
    elif phase == 'node':
        run_node_training()
    
    elif phase == 'resnet':
        run_resnet_training()
        
    elif phase == 'shap':
        trainer = ModelTrainer("data/processed/feature_dataset.csv")
        trainer.load_data()
        X, y, _ = trainer.prepare_features()
        trainer.split_data(X, y)
        
        explainer = SHAPExplainer()
        explainer.run_pipeline(X_test=trainer.X_test, y_test=trainer.y_test)
    
    elif phase == 'comparison':
        create_comprehensive_comparison()
        print_model_summary()
    
    elif phase == 'predictions':
        run_advanced_predictions()
        
    else:
        print(f"❌ Unknown phase: {phase}")
        print("Available phases: preprocessing, features, training, dnn, cnn, transformer, node, resnet, shap, comparison, predictions")


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
  python main.py --phase dnn
  python main.py --phase cnn
  python main.py --phase transformer
  python main.py --phase node
  python main.py --phase resnet
  python main.py --phase shap
  python main.py --phase comparison
  python main.py --phase predictions
  
  # Specify custom data path
  python main.py --data path/to/dataset.csv
        """
    )
    
    parser.add_argument(
        '--phase',
        type=str,
        choices=['all', 'preprocessing', 'features', 'training', 'dnn', 'cnn', 
                'transformer', 'node', 'resnet', 'shap', 'comparison', 'predictions'],
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

## 5. Final Complete Project Structure
```
"""Wearable-Data-Fusion/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── data/
│   ├── raw/
│   │   ├── .gitkeep
│   │   └── Smart Healthcare - Daily Lifestyle Dataset (Wearable device).csv
│   ├── cleaned/
│   │   ├── .gitkeep
│   │   └── canonical_dataset.csv
│   └── processed/
│       ├── .gitkeep
│       ├── feature_dataset.csv
│       └── feature_dataset_with_cluster.csv
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── dnn_training.py
│   ├── cnn_training.py
│   ├── transformer_training.py
│   ├── node_training.py
│   ├── resnet_training.py
│   ├── shap_explainability.py
│   ├── advanced_predictions.py
│   ├── model_comparison.py
│   └── utils.py
├── models/
│   ├── .gitkeep
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── lightgbm.pkl
│   ├── ensemble.pkl
│   ├── dnn_model.h5
│   ├── dnn_scaler.pkl
│   ├── cnn_model.h5
│   ├── cnn_scaler.pkl
│   ├── transformer_model.h5
│   ├── transformer_scaler.pkl
│   ├── node_model.h5
│   ├── node_scaler.pkl
│   ├── resnet_model.h5
│   └── resnet_scaler.pkl
└── outputs/
    ├── correlation_heatmap.png
    ├── dnn_training_history.png
    ├── cnn_training_history.png
    ├── transformer_training_history.png
    ├── node_training_history.png
    ├── resnet_training_history.png
    ├── shap_rf_summary.png
    ├── shap_rf_detailed.png
    ├── shap_xgb_summary.png
    ├── shap_xgb_detailed.png
    ├── shap_lgbm_summary.png
    ├── shap_lgbm_detailed.png
    ├── shap_ensemble_summary.png
    ├── shap_ensemble_detailed.png
    ├── model_comparison.png
    ├── all_models_comparison.png
    ├── shap_sleep_quality.png
    ├── shap_cv_risk.png
    ├── shap_stress_level.png
    ├── shap_stress_waterfall.png
    └── advanced_predictions_summary.png"""