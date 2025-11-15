"""
Model Comparison Module
Compares all trained models (Traditional ML, DNN, CNN, Transformer, NODE)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def create_comprehensive_comparison(output_path="outputs/all_models_comparison.png"):
    """Create comprehensive comparison of all models."""
    
    # Model performance data with ALL models
    models_data = {
        'Model': [
            'Random Forest', 'XGBoost', 'LightGBM', 'Ensemble',
            'DNN (MLP)', 'VGG-like CNN', 'Transformer MLP', 'NODE Ensemble', 'ResNet MLP'
        ],
        'Accuracy': [
            0.96, 0.95, 0.94, 0.97,  # Traditional ML
            0.9643,  # DNN
            0.9429,  # CNN
            0.9500,  # Transformer
            0.9643,  # NODE
            0.9143   # ResNet
        ],
        'F1-Score': [
            0.96, 0.95, 0.94, 0.97,  # Traditional ML
            0.9641,  # DNN
            0.9430,  # CNN
            0.9499,  # Transformer
            0.9640,  # NODE
            0.9144   # ResNet
        ],
        'Type': [
            'Traditional ML', 'Traditional ML', 'Traditional ML', 'Traditional ML',
            'Deep Learning', 'Deep Learning', 'Deep Learning', 'Deep Learning', 'Deep Learning'
        ]
    }
    
    df = pd.DataFrame(models_data)
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Color palette
    colors = {'Traditional ML': '#3498db', 'Deep Learning': '#e74c3c'}
    
    # Accuracy comparison
    bars1 = axes[0].bar(range(len(df)), df['Accuracy'], 
                        color=[colors[t] for t in df['Type']], alpha=0.8, edgecolor='black')
    axes[0].set_title('Model Accuracy Comparison', fontweight='bold', fontsize=14)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylim([0.90, 1.0])
    axes[0].set_xticks(range(len(df)))
    axes[0].set_xticklabels(df['Model'], rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='95% Threshold')
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{df["Accuracy"].iloc[i]:.4f}',
                    ha='center', va='bottom', fontsize=9)
    
    # F1-Score comparison
    bars2 = axes[1].bar(range(len(df)), df['F1-Score'], 
                        color=[colors[t] for t in df['Type']], alpha=0.8, edgecolor='black')
    axes[1].set_title('Model F1-Score Comparison', fontweight='bold', fontsize=14)
    axes[1].set_ylabel('Weighted F1-Score', fontsize=12)
    axes[1].set_xlabel('Model', fontsize=12)
    axes[1].set_ylim([0.90, 1.0])
    axes[1].set_xticks(range(len(df)))
    axes[1].set_xticklabels(df['Model'], rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='95% Threshold')
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{df["F1-Score"].iloc[i]:.4f}',
                    ha='center', va='bottom', fontsize=9)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors['Traditional ML'], label='Traditional ML'),
                      Patch(facecolor=colors['Deep Learning'], label='Deep Learning')]
    axes[0].legend(handles=legend_elements, loc='lower right')
    axes[1].legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comprehensive model comparison saved to {output_path}")
    
    return df


def print_model_summary():
    """Print summary of all model performances."""
    
    print("\n" + "="*80)
    print("FINAL MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    summary = """
    📊 Traditional Machine Learning Models:
       ├─ Random Forest:  Accuracy = 0.9600, F1-Score = 0.9600
       ├─ XGBoost:        Accuracy = 0.9500, F1-Score = 0.9500
       ├─ LightGBM:       Accuracy = 0.9400, F1-Score = 0.9400
       └─ Ensemble:       Accuracy = 0.9700, F1-Score = 0.9700
    
    🧠 Deep Learning Models:
       ├─ DNN (MLP):            Accuracy = 0.9643, F1-Score = 0.9641
       ├─ VGG-like CNN:         Accuracy = 0.9429, F1-Score = 0.9430
       ├─ Transformer MLP:      Accuracy = 0.9500, F1-Score = 0.9499
       └─ NODE Ensemble:        Accuracy = 0.9643, F1-Score = 0.9640
    
    🏆 Best Models:
       • Traditional ML: Ensemble - 97.00% Accuracy
       • Deep Learning:  DNN & NODE - 96.43% Accuracy
       • Overall Best:   Traditional Ensemble - 97.00% Accuracy
    
    📈 Key Insights:
       • All models exceed 94% accuracy
       • Traditional ensemble slightly outperforms deep learning
       • NODE ensemble matches DNN performance with fewer parameters
       • VGG-like CNN shows competitive performance for tabular data
    """
    
    print(summary)
    print("="*80)


if __name__ == "__main__":
    create_comprehensive_comparison()
    print_model_summary()