"""
Deep Neural Network Training Module
Phase 2B: Training DNN models for risk classification with standardization and LR scheduling
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import joblib
import os
import math
import warnings
warnings.filterwarnings('ignore')


class DNNTrainer:
    """Trains Deep Neural Network models for health risk classification."""
    
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize the DNN trainer.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.history = None
        self.scaler = StandardScaler()
        
    def scheduler(self, epoch, lr):
        """
        Learning rate scheduler to gradually reduce learning rate.
        
        Args:
            epoch: Current epoch number
            lr: Current learning rate
            
        Returns:
            Updated learning rate
        """
        if epoch < 10:
            return lr
        elif epoch < 30:
            return lr * math.exp(-0.1)
        else:
            return lr * math.exp(-0.05)
    
    def prepare_data(self):
        """Standardize features and encode labels."""
        print("Preparing data for DNN training...")
        
        # Standardize features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # One-hot encode labels
        self.num_classes = self.y_train.nunique()
        self.y_train_ohe = to_categorical(self.y_train, num_classes=self.num_classes)
        self.y_test_ohe = to_categorical(self.y_test, num_classes=self.num_classes)
        
        print(f"✅ Data prepared - Features: {self.X_train_scaled.shape[1]}, Classes: {self.num_classes}")
        
    def build_model(self):
        """Build the DNN architecture."""
        print("Building DNN model...")
        
        num_features = self.X_train_scaled.shape[1]
        
        self.model = Sequential([
            Dense(256, activation='relu', input_shape=(num_features,), name='h1'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu', name='h2'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu', name='h3'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(self.num_classes, activation='softmax', name='output')
        ], name='Standardized_MLP')
        
        print("✅ Model architecture created")
        print(self.model.summary())
        
    def train_model(self, epochs=150, batch_size=32, initial_lr=0.001):
        """
        Train the DNN model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            initial_lr: Initial learning rate
        """
        print(f"\nTraining DNN model (epochs={epochs}, batch_size={batch_size})...")
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Define callbacks
        callbacks_list = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            LearningRateScheduler(self.scheduler)
        ]
        
        # Train model
        self.history = self.model.fit(
            self.X_train_scaled,
            self.y_train_ohe,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_data=(self.X_test_scaled, self.y_test_ohe),
            callbacks=callbacks_list
        )
        
        print("✅ Training completed")
        
    def evaluate_model(self):
        """Evaluate the trained model."""
        print("\nEvaluating DNN model...")
        
        # Predictions
        y_prob = self.model.predict(self.X_test_scaled)
        y_pred = np.argmax(y_prob, axis=1)
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print("\n=== DNN with Standardization & LR Schedule Results ===")
        print(classification_report(self.y_test, y_pred))
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_prob
        }
    
    def plot_training_history(self, output_path="outputs/dnn_training_history.png"):
        """Plot training history."""
        print("\nPlotting training history...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('DNN Model Accuracy', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('DNN Model Loss', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training history plot saved to {output_path}")
    
    def save_model(self, model_path="models/dnn_model.h5", scaler_path="models/dnn_scaler.pkl"):
        """Save the trained model and scaler."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        self.model.save(model_path)
        print(f"✅ Model saved to {model_path}")
        
        # Save scaler
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Scaler saved to {scaler_path}")
    
    def run_pipeline(self):
        """Run the complete DNN training pipeline."""
        print("\n" + "="*60)
        print("PHASE 2B: DNN TRAINING")
        print("="*60 + "\n")
        
        self.prepare_data()
        self.build_model()
        self.train_model()
        results = self.evaluate_model()
        self.plot_training_history()
        self.save_model()
        
        print("\n✅ DNN training pipeline completed successfully!")
        return results


def main():
    """Main execution function."""
    from model_training import ModelTrainer
    
    # Load data and prepare train/test split
    print("Loading data for DNN training...")
    trainer = ModelTrainer("data/processed/feature_dataset.csv")
    trainer.load_data()
    X, y, _ = trainer.prepare_features()
    trainer.split_data(X, y)
    
    # Train DNN
    dnn_trainer = DNNTrainer(
        X_train=trainer.X_train,
        X_test=trainer.X_test,
        y_train=trainer.y_train,
        y_test=trainer.y_test
    )
    
    results = dnn_trainer.run_pipeline()
    
    print(f"\n✅ DNN training completed!")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   F1-Score: {results['f1_score']:.4f}")


if __name__ == "__main__":
    main()