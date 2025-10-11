"""
Data Preprocessing Module
Phase 0: Data Cleaning and Canonical Form Creation
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


class DataPreprocessor:
    """Handles data cleaning and transformation for wearable health data."""
    
    def __init__(self, raw_data_path: str):
        """
        Initialize the preprocessor.
        
        Args:
            raw_data_path: Path to the raw CSV dataset
        """
        self.raw_data_path = raw_data_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the raw dataset."""
        print(f"Loading data from {self.raw_data_path}...")
        self.df = pd.read_csv(self.raw_data_path)
        print(f"Original dataset shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        return self.df
    
    def rename_columns(self) -> pd.DataFrame:
        """Standardize column names for consistency."""
        print("Renaming columns...")
        self.df = self.df.rename(columns={
            "ID": "user_id",
            "Day": "day",
            "Step Count": "steps",
            "Distance Travel (Km)": "distance_km",
            "Heart Rate (BPM)": "heart_rate",
            "Blood Oxygen Level": "spO2",
            "Sleep Duration (minutes)": "sleep_min",
            "Screen Time (minute)": "screen_min",
            "Earphone Time (minute)": "earphone_min",
            "Gender": "gender"
        })
        return self.df
    
    def split_blood_pressure(self) -> pd.DataFrame:
        """Split blood pressure into systolic and diastolic components."""
        print("Splitting blood pressure...")
        bp_split = self.df["Blood Pressure"].str.split("/", expand=True)
        self.df["systolic_bp"] = pd.to_numeric(bp_split[0], errors='coerce')
        self.df["diastolic_bp"] = pd.to_numeric(bp_split[1], errors='coerce')
        self.df.drop(columns=["Blood Pressure"], inplace=True)
        return self.df
    
    def convert_data_types(self) -> pd.DataFrame:
        """Convert columns to appropriate data types."""
        print("Converting data types...")
        # Convert gender to category
        self.df["gender"] = self.df["gender"].astype("category")
        self.df["day"] = pd.to_numeric(self.df["day"], errors="coerce")
        
        # Numeric columns
        numeric_cols = [
            "Age (years)", "Height (meter)", "Weight (kg)", "BMI",
            "steps", "distance_km", "heart_rate", "spO2",
            "sleep_min", "screen_min", "earphone_min",
            "systolic_bp", "diastolic_bp"
        ]
        
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        
        return self.df
    
    def handle_missing_values(self) -> pd.DataFrame:
        """Handle missing values using median imputation."""
        print("Handling missing values...")
        
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Fill numeric columns with median
        for col in numeric_cols:
            if self.df[col].isnull().any():
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # Fill missing gender
        if "gender" in self.df.columns:
            self.df["gender"] = self.df["gender"].cat.add_categories("Unknown").fillna("Unknown")
        
        return self.df
    
    def add_basic_features(self) -> pd.DataFrame:
        """Add basic engineered features."""
        print("Adding basic features...")
        # Activity ratio
        self.df["activity_ratio"] = self.df["steps"] / self.df["distance_km"].replace(0, 1)
        
        # Sleep efficiency (fraction of day spent sleeping)
        self.df["sleep_efficiency"] = self.df["sleep_min"] / (24 * 60)
        
        return self.df
    
    def save_cleaned_data(self, output_path: str = "data/cleaned/canonical_dataset.csv"):
        """Save the cleaned dataset."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        print(f"✅ Cleaned dataset saved to {output_path}")
        print(f"Final shape: {self.df.shape}")
    
    def run_pipeline(self, output_path: str = "data/cleaned/canonical_dataset.csv") -> pd.DataFrame:
        """Run the complete preprocessing pipeline."""
        print("\n" + "="*60)
        print("PHASE 0: DATA PREPROCESSING")
        print("="*60 + "\n")
        
        self.load_data()
        self.rename_columns()
        self.split_blood_pressure()
        self.convert_data_types()
        self.handle_missing_values()
        self.add_basic_features()
        self.save_cleaned_data(output_path)
        
        print("\n✅ Preprocessing pipeline completed successfully!")
        return self.df


def main():
    """Main execution function."""
    # Path to raw data
    raw_data_path = "data/raw/SmartHealthcare_Dataset.csv"
    
    # Initialize and run preprocessor
    preprocessor = DataPreprocessor(raw_data_path)
    df_cleaned = preprocessor.run_pipeline()
    
    # Display summary
    print("\n===== Dataset Summary =====")
    print(df_cleaned.info())
    print("\n", df_cleaned.describe())


if __name__ == "__main__":
    main()