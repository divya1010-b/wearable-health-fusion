"""
Feature Engineering Module
Phase 1: Creating time-series features, delta features, and labels
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
import os


class FeatureEngineer:
    """Creates advanced features from cleaned wearable health data."""
    
    def __init__(self, cleaned_data_path: str):
        """
        Initialize the feature engineer.
        
        Args:
            cleaned_data_path: Path to the cleaned dataset
        """
        self.cleaned_data_path = cleaned_data_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the cleaned dataset."""
        print(f"Loading cleaned data from {self.cleaned_data_path}...")
        self.df = pd.read_csv(self.cleaned_data_path)
        print(f"Dataset shape: {self.df.shape}")
        return self.df
    
    def sort_data(self) -> pd.DataFrame:
        """Sort data by user and day for time-series operations."""
        print("Sorting data by user_id and day...")
        self.df = self.df.sort_values(["user_id", "day"]).reset_index(drop=True)
        return self.df
    
    def create_delta_features(self) -> pd.DataFrame:
        """Create delta (difference from previous day) features per user."""
        print("Creating delta features...")
        
        delta_cols = [
            "steps", "distance_km", "heart_rate", "spO2",
            "sleep_min", "screen_min", "earphone_min"
        ]
        
        for col in delta_cols:
            self.df[f"{col}_delta"] = self.df.groupby("user_id")[col].diff().fillna(0)
        
        return self.df
    
    def create_rolling_features(self, window: int = 3) -> pd.DataFrame:
        """
        Create rolling mean and std features per user.
        
        Args:
            window: Rolling window size (default 3 days)
        """
        print(f"Creating rolling features (window={window})...")
        
        rolling_cols = [
            "steps", "distance_km", "heart_rate", "spO2",
            "sleep_min", "screen_min", "earphone_min"
        ]
        
        for col in rolling_cols:
            # Rolling mean
            self.df[f"{col}_roll_mean"] = (
                self.df.groupby("user_id")[col]
                .rolling(window)
                .mean()
                .reset_index(0, drop=True)
            )
            
            # Rolling std
            self.df[f"{col}_roll_std"] = (
                self.df.groupby("user_id")[col]
                .rolling(window)
                .std()
                .reset_index(0, drop=True)
            )
        
        # Fill NaN values for early days
        self.df.fillna(0, inplace=True)
        
        return self.df
    
    def create_anomaly_labels(self) -> pd.DataFrame:
        """Create binary anomaly labels based on health thresholds."""
        print("Creating anomaly labels...")
        
        def is_anomaly(row):
            # Check for abnormal steps
            if row["steps"] < 2000 or row["steps"] > 20000:
                return 1
            # Check for abnormal sleep
            if row["sleep_min"] < 300 or row["sleep_min"] > 600:
                return 1
            # Check for abnormal heart rate
            if row["heart_rate"] < 50 or row["heart_rate"] > 120:
                return 1
            # Check for low SpO2
            if row["spO2"] < 92:
                return 1
            return 0
        
        self.df["Anomaly"] = self.df.apply(is_anomaly, axis=1)
        
        print(f"Anomaly distribution:\n{self.df['Anomaly'].value_counts()}")
        return self.df
    
    def create_health_score(self) -> pd.DataFrame:
        """Create a composite health score from normalized features."""
        print("Creating composite health score...")
        
        score_cols = ["steps", "distance_km", "sleep_min", "heart_rate", "spO2"]
        
        # Normalize features
        scaler = MinMaxScaler()
        self.df[score_cols] = scaler.fit_transform(self.df[score_cols])
        
        # Calculate mean as health score
        self.df["Health_Score"] = self.df[score_cols].mean(axis=1)
        
        return self.df
    
    def create_risk_levels_percentile(self) -> pd.DataFrame:
        """Create risk levels based on percentile thresholds."""
        print("Creating percentile-based risk levels...")
        
        # Compute percentile thresholds
        steps_low, steps_high = self.df["steps"].quantile([0.05, 0.95])
        sleep_low, sleep_high = self.df["sleep_min"].quantile([0.05, 0.95])
        hr_low, hr_high = self.df["heart_rate"].quantile([0.05, 0.95])
        spo2_low = self.df["spO2"].quantile(0.05)
        
        def calculate_risk(row):
            abnormal_count = 0
            
            if not (steps_low <= row["steps"] <= steps_high):
                abnormal_count += 1
            if not (sleep_low <= row["sleep_min"] <= sleep_high):
                abnormal_count += 1
            if not (hr_low <= row["heart_rate"] <= hr_high):
                abnormal_count += 1
            if row["spO2"] < spo2_low:
                abnormal_count += 1
            
            # Risk levels: 0=Low, 1=Medium, 2=High
            if abnormal_count == 0:
                return 0
            elif abnormal_count == 1:
                return 1
            else:
                return 2
        
        self.df["Risk_Level"] = self.df.apply(calculate_risk, axis=1)
        
        print(f"Risk_Level distribution:\n{self.df['Risk_Level'].value_counts()}")
        return self.df
    
    def create_risk_levels_clustering(self) -> pd.DataFrame:
        """Create risk levels using K-Means clustering."""
        print("Creating cluster-based risk levels...")
        
        cluster_features = ["steps", "sleep_min", "heart_rate", "spO2"]
        
        # Standardize features
        scaler = StandardScaler()
        X_cluster = scaler.fit_transform(self.df[cluster_features])
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        self.df["Risk_Level_Cluster"] = kmeans.fit_predict(X_cluster)
        
        # Map clusters to risk levels based on Health_Score
        cluster_order = (
            self.df.groupby("Risk_Level_Cluster")["Health_Score"]
            .mean()
            .sort_values()
            .index
        )
        
        cluster_map = {
            cluster_order[0]: 2,  # Lowest health score → High risk
            cluster_order[1]: 1,  # Medium health score → Medium risk
            cluster_order[2]: 0   # Highest health score → Low risk
        }
        
        self.df["Risk_Level_Cluster"] = self.df["Risk_Level_Cluster"].map(cluster_map)
        
        print(f"Risk_Level_Cluster distribution:\n{self.df['Risk_Level_Cluster'].value_counts()}")
        return self.df
    
    def save_features(self, output_path: str = "data/processed/feature_dataset.csv"):
        """Save the feature-engineered dataset."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"✅ Feature dataset saved to {output_path}")
        print(f"Total features: {len(self.df.columns)}")
    
    def run_pipeline(
        self,
        output_path: str = "data/processed/feature_dataset.csv"
    ) -> pd.DataFrame:
        """Run the complete feature engineering pipeline."""
        print("\n" + "="*60)
        print("PHASE 1: FEATURE ENGINEERING")
        print("="*60 + "\n")
        
        self.load_data()
        self.sort_data()
        self.create_delta_features()
        self.create_rolling_features()
        self.create_anomaly_labels()
        self.create_health_score()
        self.create_risk_levels_percentile()
        self.create_risk_levels_clustering()
        self.save_features(output_path)
        
        print("\n✅ Feature engineering completed successfully!")
        return self.df


def main():
    """Main execution function."""
    # Path to cleaned data
    cleaned_data_path = "data/cleaned/canonical_dataset.csv"
    
    # Initialize and run feature engineer
    engineer = FeatureEngineer(cleaned_data_path)
    df_features = engineer.run_pipeline()
    
    # Display summary
    print("\n===== Feature Dataset Summary =====")
    print(f"Shape: {df_features.shape}")
    print(f"\nColumns: {df_features.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df_features.head())


if __name__ == "__main__":
    main()