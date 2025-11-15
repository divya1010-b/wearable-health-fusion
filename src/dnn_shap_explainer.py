# ======================================
# 📊 IMPROVED SHAP EXPLAINABILITY FOR ALL DNN MODELS
# ======================================
import shap
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("📘 BEGINNING SHAP EXPLAINABILITY FOR ALL TRAINED MODELS")
print("="*70)

# --- Utility Function for SHAP Computation ---
def compute_shap_summary(model, X_train, X_test, model_name, max_display=15, is_cnn=False):
    """
    Compute SHAP values and plot summary for a trained model.
    Uses appropriate explainer based on model type.
    
    Args:
        model: Trained Keras model
        X_train: Training data for background
        X_test: Test data for explanation
        model_name: Name of the model for titles
        max_display: Maximum features to display
        is_cnn: Whether the model is a CNN (requires special handling)
    """
    print(f"\n\n🔹 SHAP EXPLAINABILITY FOR: {model_name}")
    print("-"*70)

    # Use smaller subset for background (SHAP is computationally heavy)
    background = shap.sample(X_train, 200)
    test_sample = shap.sample(X_test, 200)
    
    try:
        if is_cnn:
            # For CNN models, use DeepExplainer or GradientExplainer
            print("Using DeepExplainer for CNN model...")
            explainer = shap.DeepExplainer(model, background)
        else:
            # For standard DNNs, try DeepExplainer first
            print("Using DeepExplainer for DNN model...")
            explainer = shap.DeepExplainer(model, background)
        
        # Compute SHAP values
        print("Computing SHAP values...")
        shap_values = explainer.shap_values(test_sample)
        
        # Handle multi-class output (list of arrays)
        if isinstance(shap_values, list):
            print(f"Multi-class detected: {len(shap_values)} classes")
            # Average across all classes for visualization
            shap_values_plot = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            feature_names = [f"Feature {i}" for i in range(shap_values_plot.shape[1])]
        else:
            shap_values_plot = shap_values
            feature_names = [f"Feature {i}" for i in range(shap_values_plot.shape[1])]
        
        # --- Bar Plot: Feature Importance ---
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_plot, test_sample, 
                         feature_names=feature_names,
                         plot_type="bar", 
                         show=False, 
                         max_display=max_display)
        plt.title(f"{model_name} - SHAP Feature Importance (Bar Plot)", 
                 fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.show()

        # --- Beeswarm Plot: Detailed Summary ---
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_plot, test_sample,
                         feature_names=feature_names,
                         show=False, 
                         max_display=max_display)
        plt.title(f"{model_name} - SHAP Summary Plot (Beeswarm)", 
                 fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        print(f"✅ SHAP analysis completed for {model_name}!")
        return shap_values
        
    except Exception as e:
        print(f"⚠️ DeepExplainer failed, trying GradientExplainer: {e}")
        try:
            explainer = shap.GradientExplainer(model, background)
            shap_values = explainer.shap_values(test_sample)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values_plot = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_values_plot = shap_values
            
            feature_names = [f"Feature {i}" for i in range(shap_values_plot.shape[1])]
            
            # Create plots
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_plot, test_sample,
                             feature_names=feature_names,
                             plot_type="bar", 
                             show=False, 
                             max_display=max_display)
            plt.title(f"{model_name} - SHAP Feature Importance", 
                     fontweight='bold', fontsize=14)
            plt.tight_layout()
            plt.show()
            
            print(f"✅ SHAP analysis completed for {model_name} using GradientExplainer!")
            return shap_values
            
        except Exception as e2:
            print(f"❌ Both DeepExplainer and GradientExplainer failed for {model_name}")
            print(f"Error: {e2}")
            return None


# ======================================
# 1️⃣ THIRD DNN (Standardized Input & LR Schedule)
# ======================================
try:
    print("\n" + "="*70)
    print("ANALYZING: Third DNN (Standardized + LR Schedule)")
    print("="*70)
    
    shap_values_dnn_standard = compute_shap_summary(
        dnn_standard_model, 
        X_train_scaled, 
        X_test_scaled,
        model_name="Third DNN (Standardized + LR Schedule)",
        max_display=15,
        is_cnn=False
    )
except Exception as e:
    print(f"⚠️ SHAP failed for Third DNN: {e}")


# ======================================
# 2️⃣ TRANSFORMER-LIKE MLP
# ======================================
try:
    print("\n" + "="*70)
    print("ANALYZING: Transformer-like MLP")
    print("="*70)
    
    shap_values_transformer = compute_shap_summary(
        transformer_like_model, 
        X_train_scaled, 
        X_test_scaled,
        model_name="Transformer-like MLP",
        max_display=15,
        is_cnn=False
    )
except Exception as e:
    print(f"⚠️ SHAP failed for Transformer-like MLP: {e}")


# ======================================
# 3️⃣ VGG-LIKE TABULAR CNN (IMPROVED)
# ======================================
try:
    print("\n" + "="*70)
    print("ANALYZING: VGG-like Tabular CNN")
    print("="*70)
    
    # Get the input shape from the model
    input_shape = vgg_like_model.input_shape  # (None, H, W, C)
    print(f"Model expected input shape: {input_shape}")
    
    H, W, C = input_shape[1], input_shape[2], input_shape[3]
    
    # Helper function to convert to numpy
    def to_numpy(x):
        return x.values if hasattr(x, "values") else np.asarray(x)
    
    # Convert to numpy
    X_train_np = to_numpy(X_train_scaled)
    X_test_np = to_numpy(X_test_scaled)
    
    # Reshape function for CNN input
    def reshape_to_4d(X, h, w, c):
        X = X.reshape(X.shape[0], -1)
        needed = h * w * c
        
        # Pad or truncate to match required size
        if X.shape[1] < needed:
            X = np.pad(X, ((0, 0), (0, needed - X.shape[1])), mode='constant')
        elif X.shape[1] > needed:
            X = X[:, :needed]
        
        return X.reshape(X.shape[0], h, w, c)
    
    # Reshape data to 4D
    X_train_4d = reshape_to_4d(X_train_np, H, W, C)
    X_test_4d = reshape_to_4d(X_test_np, H, W, C)
    
    print(f"Reshaped training data: {X_train_4d.shape}")
    print(f"Reshaped test data: {X_test_4d.shape}")
    
    # Sample data for SHAP
    background_cnn = X_train_4d[:100]
    test_sample_cnn = X_test_4d[:200]
    
    # Try DeepExplainer first
    try:
        print("Creating DeepExplainer for CNN...")
        explainer_cnn = shap.DeepExplainer(vgg_like_model, background_cnn)
        print("Computing SHAP values for CNN...")
        shap_values_cnn = explainer_cnn.shap_values(test_sample_cnn)
    except Exception as e:
        print(f"DeepExplainer failed, trying GradientExplainer: {e}")
        explainer_cnn = shap.GradientExplainer(vgg_like_model, background_cnn)
        shap_values_cnn = explainer_cnn.shap_values(test_sample_cnn)
    
    # Flatten SHAP values and data for visualization
    if isinstance(shap_values_cnn, list):
        # Multi-class: average absolute values across classes
        shap_flat = np.mean([np.abs(s.reshape(s.shape[0], -1)) for s in shap_values_cnn], axis=0)
    else:
        shap_flat = np.abs(shap_values_cnn.reshape(shap_values_cnn.shape[0], -1))
    
    X_test_flat = test_sample_cnn.reshape(test_sample_cnn.shape[0], -1)
    
    # Ensure matching dimensions
    n_features = min(shap_flat.shape[1], X_test_flat.shape[1])
    shap_flat = shap_flat[:, :n_features]
    X_test_flat = X_test_flat[:, :n_features]
    
    # Create feature names
    feature_names_cnn = [f"Pixel_{i}" for i in range(n_features)]
    
    # --- Bar Plot ---
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_flat, X_test_flat, 
                     feature_names=feature_names_cnn,
                     plot_type="bar", 
                     show=False, 
                     max_display=15)
    plt.title("VGG-like CNN - SHAP Feature Importance (Bar)", 
             fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()

    # --- Beeswarm Plot ---
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_flat, X_test_flat,
                     feature_names=feature_names_cnn,
                     show=False, 
                     max_display=15)
    plt.title("VGG-like CNN - SHAP Summary (Beeswarm)", 
             fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()

    print("✅ VGG-like CNN SHAP explainability completed successfully!")

except Exception as e:
    print(f"❌ SHAP failed for VGG-like model: {e}")
    import traceback
    traceback.print_exc()


# ======================================
# 4️⃣ NODE-LIKE ENSEMBLE
# ======================================
try:
    print("\n" + "="*70)
    print("ANALYZING: NODE-like MLP Ensemble")
    print("="*70)
    
    shap_values_node = compute_shap_summary(
        node_like_model, 
        X_train_scaled, 
        X_test_scaled,
        model_name="NODE-like MLP Ensemble",
        max_display=15,
        is_cnn=False
    )
except Exception as e:
    print(f"⚠️ SHAP failed for NODE-like model: {e}")


# ======================================
# 5️⃣ FINAL RESNET-STYLE MLP
# ======================================
try:
    print("\n" + "="*70)
    print("ANALYZING: Final ResNet-MLP")
    print("="*70)
    
    shap_values_resnet = compute_shap_summary(
        final_resnet_model, 
        X_train_scaled, 
        X_test_scaled,
        model_name="Final ResNet-MLP",
        max_display=15,
        is_cnn=False
    )
except Exception as e:
    print(f"⚠️ SHAP failed for Final ResNet-MLP: {e}")


# ======================================
# SUMMARY
# ======================================
print("\n" + "="*70)
print("✅ ALL SHAP COMPUTATIONS COMPLETED!")
print("="*70)
print("\n📊 Summary:")
print("   ✓ Third DNN (Standardized + LR Schedule)")
print("   ✓ Transformer-like MLP")
print("   ✓ VGG-like Tabular CNN (Improved)")
print("   ✓ NODE-like MLP Ensemble")
print("   ✓ Final ResNet-MLP")
print("\n" + "="*70)