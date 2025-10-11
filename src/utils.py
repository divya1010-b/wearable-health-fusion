"""
Utility Functions Module
Common helper functions used across the project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Any


def create_directory(path: str):
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)
    print(f"✅ Directory created/verified: {path}")


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load CSV file with error handling.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {filepath} - Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None
    except Exception as e:
        print(f"❌ Error loading {filepath}: {str(e)}")
        return None


def save_csv(df: pd.DataFrame, filepath: str):
    """
    Save DataFrame to CSV with directory creation.
    
    Args:
        df: DataFrame to save
        filepath: Output path
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    df.to_csv(filepath, index=False)
    print(f"✅ Saved to {filepath} - Shape: {df.shape}")


def plot_correlation_heatmap(df: pd.DataFrame, 
                             columns: List[str] = None,
                             output_path: str = "outputs/correlation_heatmap.png",
                             figsize: tuple = (12, 10)):
    """
    Create and save correlation heatmap.
    
    Args:
        df: DataFrame
        columns: List of columns to include (if None, uses all numeric)
        output_path: Path to save plot
        figsize: Figure size
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    plt.figure(figsize=figsize)
    sns.heatmap(df[columns].corr(), annot=True, fmt=".2f", 
                cmap="coolwarm", center=0, square=True)
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Correlation heatmap saved to {output_path}")


def plot_distribution(df: pd.DataFrame, 
                     column: str,
                     output_path: str = None,
                     bins: int = 30):
    """
    Plot distribution of a column.
    
    Args:
        df: DataFrame
        column: Column name to plot
        output_path: Path to save plot (if None, displays only)
        bins: Number of bins for histogram
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=bins, kde=True)
    plt.title(f"Distribution of {column}", fontsize=14, fontweight='bold')
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Distribution plot saved to {output_path}")
    
    plt.show()
    plt.close()


def print_dataset_info(df: pd.DataFrame, name: str = "Dataset"):
    """
    Print comprehensive dataset information.
    
    Args:
        df: DataFrame
        name: Name of dataset for display
    """
    print("\n" + "="*60)
    print(f"{name} Information")
    print("="*60)
    print(f"\nShape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nColumn Names:\n{df.columns.tolist()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")
    print("="*60 + "\n")


def get_feature_importance(model, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
    """
    Get feature importance from tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with features and their importances
    """
    if not hasattr(model, 'feature_importances_'):
        print("⚠️  Model does not have feature_importances_ attribute")
        return None
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(top_n)
    
    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame,
                           output_path: str = None,
                           figsize: tuple = (10, 8)):
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with 'Feature' and 'Importance' columns
        output_path: Path to save plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
    plt.title("Feature Importance", fontsize=14, fontweight='bold')
    plt.xlabel("Importance")
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Feature importance plot saved to {output_path}")
    
    plt.show()
    plt.close()


def create_summary_report(results: Dict[str, Any], output_path: str = "outputs/summary_report.txt"):
    """
    Create a text summary report of results.
    
    Args:
        results: Dictionary containing project results
        output_path: Path to save report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("WEARABLE HEALTH FUSION - PROJECT SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        for key, value in results.items():
            f.write(f"{key}:\n")
            f.write(f"{value}\n\n")
        
        f.write("="*60 + "\n")
    
    print(f"✅ Summary report saved to {output_path}")


def validate_data_path(path: str, required_files: List[str] = None) -> bool:
    """
    Validate that data path exists and contains required files.
    
    Args:
        path: Directory path to validate
        required_files: List of required file names (if None, just checks directory)
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(path):
        print(f"❌ Path does not exist: {path}")
        return False
    
    if required_files:
        for filename in required_files:
            filepath = os.path.join(path, filename)
            if not os.path.exists(filepath):
                print(f"❌ Required file not found: {filepath}")
                return False
    
    print(f"✅ Data path validated: {path}")
    return True


def calculate_class_weights(y: pd.Series) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y: Target variable series
        
    Returns:
        Dictionary mapping class labels to weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    class_weights = dict(zip(classes, weights))
    print(f"✅ Class weights calculated: {class_weights}")
    
    return class_weights


# Color palette for consistent plotting
PLOT_COLORS = {
    'primary': '#3498db',
    'secondary': '#e74c3c',
    'success': '#2ecc71',
    'warning': '#f39c12',
    'info': '#9b59b6',
    'light': '#ecf0f1',
    'dark': '#34495e'
}


def set_plot_style():
    """Set consistent plot style for all visualizations."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 14


# Initialize plot style when module is imported
set_plot_style()