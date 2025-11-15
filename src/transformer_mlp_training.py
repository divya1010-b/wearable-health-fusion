"""
Transformer-like MLP Training Module
Phase 2D: Training Transformer-inspired models with residual connections
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, Add
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class TransformerTrainer:
    """Trains Transformer-like MLP models with residual connections."""
    
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize the Transformer trainer.
        
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
        
    def transformer_block(self, x, units, dropout_rate=0.15, name_prefix='block'):
        """
        Transformer-inspired block with normalization and residual connections.
        
        Args:
            x: Input tensor
            units: Number of units in dense layers
            dropout_rate: Dropout rate
            name_prefix: Prefix for layer names
            
        Returns:
            Output tensor with residual connection
        """
        # Store input for residual connection
        shortcut = x
        
        # Batch Normalization
        y = BatchNormalization(name=f'{name_prefix}_bn1')(x)
        
        # Feed-Forward Network
        y = Dense(units, activation='gelu', name=f'{name_prefix}_dense1')(y)
        y = Dropout(dropout_rate)(y)
        
        y = Dense(units, activation='gelu', name=f'{name_prefix}_dense2')(y)
        y = Dropout(dropout_rate)(y)
        
        # Residual Connection
        output = Add(name=f'{name_prefix}_add')([shortcut, y])
        
        return output
    
    def prepare_data(self):
        """Standardize features and encode labels."""
        print("Preparing data for Transformer-like MLP training...")
        
        # Standardize features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # One-hot encode labels
        self.num_classes = self.y_train.nunique()
        self.y_train_ohe = to_categorical(self.y_train, num_classes=self.num_classes)
        self.y_test_ohe = to_categorical(self.y_test, num_classes=self.num_classes)
        
        print(f"✅ Data prepared - Features: {self.X_train_scaled.shape[1]}, Classes: {self.num_classes}")
    
    def build_model(self, embedding_dim=256, num_blocks=6):
        """
        Build Transformer-like MLP architecture.
        
        Args:
            embedding_dim: Dimension of embedding space
            num_blocks: Number of transformer blocks
        """
        print("Building Transformer-like MLP model...")
        
        num_features = self.X_train_scaled.shape[1]
        
        # Input layer
        inputs = Input(shape=(num_features,), name="input_features")
        
        # Feature embedding
        x = Dense(embedding_dim, activation='gelu', name='feature_embed')(inputs)
        
        # Stack Transformer blocks
        for i in range(num_blocks):
            x = self.transformer_block(x, embedding_dim, dropout_rate=0.15, name_prefix=f'block_{i}')
        
        # Final classification head
        x = BatchNormalization(name='final_bn')(x)
        x = Dropout(0.2)(x)
        
        outputs = Dense(self.num_classes, activation='softmax', name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='Tabular_Transformer_Like')
        
        print("✅ Transformer-like MLP architecture created")
        print(self.model.summary())
    
    def train_model(self, epochs=250, batch_size=64, initial_lr=0.0008):
        """
        Train the Transformer-like MLP model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            initial_lr: Initial learning rate
        """
        print(f"\nTraining Transformer-like MLP (epochs={epochs}, batch_size={batch_size})...")
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=initial_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Define callbacks
        callbacks_list = [
            EarlyStopping(
                monitor='val_loss',
                patience=25,
                restore_best_weights=True,
                verbose=1
            )
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
        print("\nEvaluating Transformer-like MLP model...")
        
        # Predictions
        y_prob = self.model.predict(self.X_test_scaled)
        y_pred = np.argmax(y_prob, axis=1)
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print("\n=== Transformer-like MLP Results ===")
        print(classification_report(self.y_test, y_pred))
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_prob
        }
    
    def plot_training_history(self, output_path="outputs/transformer_training_history.png"):
        """Plot training history."""
        print("\nPlotting Transformer training history...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy', color='blue')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy', color='orange')
        axes[0].set_title('Transformer-like MLP Model Accuracy', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Train Loss', color='blue')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss', color='orange')
        axes[1].set_title('Transformer-like MLP Model Loss', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Transformer training history plot saved to {output_path}")
    
    def save_model(self, model_path="models/transformer_model.h5", scaler_path="models/transformer_scaler.pkl"):
        """Save the trained model and scaler."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        self.model.save(model_path)
        print(f"✅ Transformer model saved to {model_path}")
        
        # Save scaler
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Transformer scaler saved to {scaler_path}")
    
    def run_pipeline(self):
        """Run the complete Transformer training pipeline."""
        print("\n" + "="*60)
        print("PHASE 2D: TRANSFORMER-LIKE MLP TRAINING")
        print("="*60 + "\n")
        
        self.prepare_data()
        self.build_model()
        self.train_model()
        results = self.evaluate_model()
        self.plot_training_history()
        self.save_model()
        
        print("\n✅ Transformer-like MLP training pipeline completed successfully!")
        return results


def main():
    """Main execution function."""
    from model_training import ModelTrainer
    
    # Load data and prepare train/test split
    print("Loading data for Transformer training...")
    trainer = ModelTrainer("data/processed/feature_dataset.csv")
    trainer.load_data()
    X, y, _ = trainer.prepare_features()
    trainer.split_data(X, y)
    
    # Train Transformer
    transformer_trainer = TransformerTrainer(
        X_train=trainer.X_train,
        X_test=trainer.X_test,
        y_train=trainer.y_train,
        y_test=trainer.y_test
    )
    
    results = transformer_trainer.run_pipeline()
    
    print(f"\n✅ Transformer-like MLP training completed!")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   F1-Score: {results['f1_score']:.4f}")


if __name__ == "__main__":
    main()