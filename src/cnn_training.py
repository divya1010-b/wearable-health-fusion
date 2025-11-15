"""
CNN Training Module (VGG-like Architecture)
Phase 2C: Training CNN models for tabular data classification
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten
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


class CNNTrainer:
    """Trains VGG-like CNN models for tabular health risk classification."""
    
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize the CNN trainer.
        
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
        Learning rate scheduler with gradual reduction.
        
        Args:
            epoch: Current epoch number
            lr: Current learning rate
            
        Returns:
            Updated learning rate
        """
        return lr * math.exp(-0.02) if epoch > 20 else lr
    
    def prepare_data_2d(self):
        """Standardize features and reshape to 2D grid for CNN."""
        print("Preparing 2D data for CNN training...")
        
        # Standardize features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Determine 2D grid size
        num_features = self.X_train_scaled.shape[1]
        self.grid_size = math.ceil(math.sqrt(num_features))
        target_features = self.grid_size * self.grid_size
        padding_needed = target_features - num_features
        
        self.image_h = self.grid_size
        self.image_w = self.grid_size
        self.channels = 1
        
        # Pad and reshape to 2D
        def pad_and_reshape(X, padding):
            X_padded = np.pad(X, ((0, 0), (0, padding)), 'constant')
            return X_padded.reshape(X.shape[0], self.image_h, self.image_w, self.channels)
        
        self.X_train_2d = pad_and_reshape(self.X_train_scaled, padding_needed)
        self.X_test_2d = pad_and_reshape(self.X_test_scaled, padding_needed)
        
        # One-hot encode labels
        self.num_classes = self.y_train.nunique()
        self.y_train_ohe = to_categorical(self.y_train, num_classes=self.num_classes)
        self.y_test_ohe = to_categorical(self.y_test, num_classes=self.num_classes)
        
        print(f"✅ Data reshaped - Original: {num_features} features → Grid: {self.image_h}x{self.image_w}x{self.channels} (Padded to {target_features})")
        
    def build_vgg_model(self):
        """Build VGG-like architecture for tabular data."""
        print("Building VGG-like CNN model...")
        
        self.model = Sequential([
            # Block 1: 2 Conv layers
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                   input_shape=(self.image_h, self.image_w, self.channels), name='conv_1_1'),
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='conv_1_2'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2), name='max_pool_1'),
            Dropout(0.2),
            
            # Block 2: 2 Conv layers (increased filters)
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='conv_2_1'),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='conv_2_2'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2), name='max_pool_2'),
            Dropout(0.3),
            
            # Flattening
            Flatten(),
            
            # Final Classification Head
            Dense(128, activation='relu', name='dense_1'),
            BatchNormalization(),
            Dropout(0.4),
            
            # Output Layer
            Dense(self.num_classes, activation='softmax', name='output')
        ], name='VGG_Tabular_Experimental')
        
        print("✅ VGG-like CNN architecture created")
        print(self.model.summary())
        
    def train_model(self, epochs=300, batch_size=64, initial_lr=0.0005):
        """
        Train the VGG-like CNN model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            initial_lr: Initial learning rate
        """
        print(f"\nTraining VGG-like CNN (epochs={epochs}, batch_size={batch_size})...")
        
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
                patience=30,
                restore_best_weights=True,
                verbose=1
            ),
            LearningRateScheduler(self.scheduler)
        ]
        
        # Train model
        self.history = self.model.fit(
            self.X_train_2d,
            self.y_train_ohe,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_data=(self.X_test_2d, self.y_test_ohe),
            callbacks=callbacks_list
        )
        
        print("✅ Training completed")
        
    def evaluate_model(self):
        """Evaluate the trained CNN model."""
        print("\nEvaluating VGG-like CNN model...")
        
        # Predictions
        y_prob = self.model.predict(self.X_test_2d)
        y_pred = np.argmax(y_prob, axis=1)
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print("\n=== Experimental VGG-like CNN Results ===")
        print(classification_report(self.y_test, y_pred))
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_prob
        }
    
    def plot_training_history(self, output_path="outputs/cnn_training_history.png"):
        """Plot training history."""
        print("\nPlotting CNN training history...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy', color='blue')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy', color='orange')
        axes[0].set_title('VGG-like CNN Model Accuracy', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Train Loss', color='blue')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss', color='orange')
        axes[1].set_title('VGG-like CNN Model Loss', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ CNN training history plot saved to {output_path}")
    
    def save_model(self, model_path="models/cnn_model.h5", scaler_path="models/cnn_scaler.pkl"):
        """Save the trained CNN model and scaler."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        self.model.save(model_path)
        print(f"✅ CNN model saved to {model_path}")
        
        # Save scaler
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ CNN scaler saved to {scaler_path}")
    
    def run_pipeline(self):
        """Run the complete CNN training pipeline."""
        print("\n" + "="*60)
        print("PHASE 2C: VGG-LIKE CNN TRAINING")
        print("="*60 + "\n")
        
        self.prepare_data_2d()
        self.build_vgg_model()
        self.train_model()
        results = self.evaluate_model()
        self.plot_training_history()
        self.save_model()
        
        print("\n✅ VGG-like CNN training pipeline completed successfully!")
        return results


def main():
    """Main execution function."""
    from model_training import ModelTrainer
    
    # Load data and prepare train/test split
    print("Loading data for CNN training...")
    trainer = ModelTrainer("data/processed/feature_dataset.csv")
    trainer.load_data()
    X, y, _ = trainer.prepare_features()
    trainer.split_data(X, y)
    
    # Train CNN
    cnn_trainer = CNNTrainer(
        X_train=trainer.X_train,
        X_test=trainer.X_test,
        y_train=trainer.y_train,
        y_test=trainer.y_test
    )
    
    results = cnn_trainer.run_pipeline()
    
    print(f"\n✅ VGG-like CNN training completed!")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   F1-Score: {results['f1_score']:.4f}")


if __name__ == "__main__":
    main()