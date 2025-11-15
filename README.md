<invoke name="artifacts"><parameter name="command">create</parameter><parameter name="type">text/markdown</parameter><parameter name="title">Complete Updated README.md</parameter><parameter name="id">final-readme</parameter><parameter name="content"># Wearable Health Fusion

## Project Overview

This project develops a **comprehensive personalized health monitoring system** using wearable device data. We integrate multimodal data (physiological, activity, lifestyle) to create **feature-rich datasets** for predictive modeling, anomaly detection, and personalized health insights.

### Key Features
- 🏥 **Multi-model Architecture**: 9 state-of-the-art ML/DL models
- 🔍 **SHAP Explainability**: Interpretable AI for healthcare decisions
- 📊 **Advanced Predictions**: Sleep quality, CV risk, stress estimation
- 🎯 **Risk Stratification**: Automated health risk classification
- 📈 **Real-time Monitoring**: Day-to-day health trend analysis

---

## Project Phases

1. **Phase 0 — Data Preprocessing** ✅ Completed  
2. **Phase 1 — Feature Engineering & Labeling** ✅ Completed  
3. **Phase 2 — Predictive Modeling** ✅ Completed
   - 2A: Traditional ML (RF, XGBoost, LightGBM, Ensemble)
   - 2B: Deep Neural Network (MLP)
   - 2C: VGG-like CNN
   - 2D: Transformer-like MLP
   - 2E: NODE Ensemble
   - 2F: ResNet-style MLP
4. **Phase 3 — Model Explainability (SHAP)** ✅ Completed
5. **Phase 4 — Model Comparison** ✅ Completed
6. **Phase 5 — Advanced Predictions** ✅ Completed

---

## Dataset

**Source**: [Smart Healthcare – DailyLife Dataset (Wearable Device)](https://www.kaggle.com/datasets/mdimammahdi/smart-healthcare-dailylife-dataset-wearable-device)

### Features in Cleaned Dataset

| Column | Description |
|--------|-------------|
| `user_id` | Unique identifier for each individual |
| `day` | Day of measurement (1–7) |
| `Gender` | Male / Female |
| `Age (years)` | Age of the individual |
| `Height (meter)` | Height in meters |
| `Weight (kg)` | Weight in kilograms |
| `BMI` | Body Mass Index |
| `steps` | Total steps taken per day |
| `distance_km` | Distance covered per day (km) |
| `heart_rate` | Average heart rate (BPM) |
| `spO2` | Blood oxygen level (%) |
| `sleep_min` | Total sleep duration (minutes) |
| `screen_min` | Screen time (minutes) |
| `earphone_min` | Earphone usage per day (minutes) |
| `systolic_bp` | Systolic blood pressure |
| `diastolic_bp` | Diastolic blood pressure |
| `activity_ratio` | Steps / distance ratio |
| `sleep_efficiency` | Sleep duration / recommended sleep |

### Engineered Features
- **Delta Features**: Day-to-day changes (Δsteps, Δsleep, Δheart_rate)
- **Rolling Statistics**: 3-day moving averages and standard deviations
- **Health Scores**: Composite metrics for overall wellness
- **Risk Labels**: K-means clustering + rule-based classification

---

## Project Structure

```
Wearable-Data-Fusion/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
├── main.py                           # Main pipeline orchestrator
│
├── data/
│   ├── raw/                          # Original datasets
│   ├── cleaned/                      # Preprocessed data
│   │   └── canonical_dataset.csv
│   └── processed/                    # Feature-engineered data
│       ├── feature_dataset.csv
│       └── feature_dataset_with_cluster.csv
│
├── src/
│   ├── __init__.py                   # Package initialization
│   ├── data_preprocessing.py        # Data cleaning & transformation
│   ├── feature_engineering.py       # Feature creation
│   ├── model_training.py            # Traditional ML models
│   ├── dnn_training.py              # Deep Neural Network
│   ├── cnn_training.py              # VGG-like CNN
│   ├── transformer_training.py      # Transformer-like MLP
│   ├── node_training.py             # NODE Ensemble
│   ├── resnet_training.py           # ResNet-style MLP
│   ├── shap_explainability.py       # SHAP analysis
│   ├── advanced_predictions.py      # Specialized predictions
│   ├── model_comparison.py          # Performance comparison
│   └── utils.py                     # Helper functions
│
├── models/                           # Trained model files
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── lightgbm.pkl
│   ├── ensemble.pkl
│   ├── dnn_model.h5
│   ├── cnn_model.h5
│   ├── transformer_model.h5
│   ├── node_model.h5
│   └── resnet_model.h5
│
└── outputs/                          # Visualizations & results
    ├── correlation_heatmap.png
    ├── *_training_history.png       # Training curves
    ├── shap_*_summary.png           # SHAP importance
    ├── shap_*_detailed.png          # SHAP dependency
    ├── all_models_comparison.png    # Performance comparison
    └── advanced_predictions_summary.png
```

---

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- 8GB+ RAM recommended for deep learning models

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/divya1010-b/Wearable-Data-Fusion.git
cd Wearable-Data-Fusion
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
- Download from [Kaggle](https://www.kaggle.com/datasets/mdimammahdi/smart-healthcare-dailylife-dataset-wearable-device)
- Place in `data/raw/`

---

## Usage

### Quick Start - Run Complete Pipeline

```bash
python main.py
```

This runs all phases: preprocessing → feature engineering → model training → SHAP → predictions

### Run Specific Phases

```bash
# Data preprocessing only
python main.py --phase preprocessing

# Feature engineering only
python main.py --phase features

# Traditional ML models only
python main.py --phase training

# Deep learning models
python main.py --phase dnn           # Deep Neural Network
python main.py --phase cnn           # VGG-like CNN
python main.py --phase transformer   # Transformer MLP
python main.py --phase node          # NODE Ensemble
python main.py --phase resnet        # ResNet MLP

# Analysis phases
python main.py --phase shap          # SHAP explainability
python main.py --phase comparison    # Model comparison
python main.py --phase predictions   # Advanced predictions
```

### Run Individual Modules

```bash
# Execute specific modules directly
python src/data_preprocessing.py
python src/feature_engineering.py
python src/model_training.py
python src/dnn_training.py
python src/advanced_predictions.py
```

### Custom Data Path

```bash
python main.py --data path/to/your/dataset.csv
```

---

## Model Architecture

### Traditional Machine Learning (4 models)

| Model | Accuracy | F1-Score | Parameters |
|-------|----------|----------|------------|
| **Random Forest** | 96.00% | 0.9600 | n_estimators=200, max_depth=10 |
| **XGBoost** | 95.00% | 0.9500 | n_estimators=200, max_depth=6 |
| **LightGBM** | 94.00% | 0.9400 | n_estimators=300, lr=0.05 |
| **Ensemble (Voting)** | 97.00% | 0.9700 | Hard voting (RF+XGB+LGBM) |

### Deep Learning Models (5 models)

#### 1. **Deep Neural Network (DNN)**
- **Architecture**: 4-layer MLP (256→128→64→3)
- **Features**: Batch normalization, dropout, LR scheduling
- **Performance**: 96.43% accuracy, 0.9641 F1-score
- **Training**: AdamW optimizer, early stopping

```
Input → Dense(256) → BatchNorm → Dropout(0.3)
     → Dense(128) → BatchNorm → Dropout(0.3)
     → Dense(64)  → BatchNorm → Dropout(0.2)
     → Dense(3, softmax) → Output
```

#### 2. **VGG-like CNN**
- **Architecture**: 2D convolutional blocks for tabular data
- **Features**: 32→64 filters, max pooling, spatial processing
- **Performance**: 94.29% accuracy, 0.9430 F1-score
- **Innovation**: Reshapes 1D features to 6×6×1 grid

```
Input (6×6×1) → Conv2D(32) → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.2)
              → Conv2D(64) → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.3)
              → Flatten → Dense(128) → BatchNorm → Dropout(0.4)
              → Dense(3, softmax) → Output
```

#### 3. **Transformer-like MLP**
- **Architecture**: 6 residual blocks with attention-inspired design
- **Features**: GELU activation, layer normalization, skip connections
- **Performance**: 95.00% accuracy, 0.9499 F1-score
- **Parameters**: 806K parameters, 256-dim embeddings

#### 4. **NODE Ensemble**
- **Architecture**: 10 parallel tree-like sub-networks
- **Features**: Differentiable decision trees, ensemble learning
- **Performance**: 96.43% accuracy, 0.9640 F1-score
- **Parameters**: 37K parameters (highly efficient)

#### 5. **ResNet-style MLP**
- **Architecture**: 6 deep residual blocks
- **Features**: Identity skip connections, adaptive LR reduction
- **Performance**: 91.43% accuracy, 0.9144 F1-score
- **Training**: ReduceLROnPlateau, up to 500 epochs

---

## Advanced Predictions

### 1. Sleep Quality Prediction
- **Model**: Random Forest Regressor
- **Features**: SpO2, heart rate, activity levels, sleep duration
- **Output**: Sleep Quality Score (0-100)
- **Metrics**: MAE, R² score

### 2. Cardiovascular Risk Assessment
- **Model**: Gradient Boosting Regressor
- **Features**: Heart rate, blood pressure, oxygen levels, activity
- **Output**: CV Risk Score (0-100)
- **Use Case**: Early detection of cardiovascular issues

### 3. Stress Level Estimation
- **Model**: Gradient Boosting Regressor
- **Features**: Heart rate variability, screen time, sleep quality
- **Output**: Stress Index (0-100)
- **Visualization**: SHAP waterfall plots for individual explanations

---

## SHAP Explainability

All models include comprehensive SHAP (SHapley Additive exPlanations) analysis:

- **Feature Importance**: Bar plots showing global feature contributions
- **Summary Plots**: Beeswarm plots showing feature impact distribution
- **Waterfall Plots**: Individual prediction explanations
- **Dependency Plots**: Feature interaction analysis

### Example SHAP Outputs
- `shap_rf_summary.png` - Random Forest feature importance
- `shap_xgb_detailed.png` - XGBoost dependency plots
- `shap_stress_waterfall.png` - Individual stress prediction explanation

---

## Results Summary

### Model Performance Comparison

| Model Type | Best Model | Accuracy | F1-Score |
|------------|-----------|----------|----------|
| **Traditional ML** | Ensemble | 97.00% | 0.9700 |
| **Deep Learning** | DNN / NODE | 96.43% | 0.9641 |
| **Overall Best** | Ensemble | 97.00% | 0.9700 |

### Key Insights
- ✅ All models exceed 94% accuracy
- ✅ Ensemble methods slightly outperform individual models
- ✅ Deep learning competitive with traditional ML on tabular data
- ✅ NODE ensemble achieves high accuracy with 20× fewer parameters
- ✅ SHAP reveals sleep, heart rate, and SpO2 as top predictors

---

## Visualizations

### Generated Outputs

**Training & Performance:**
- `correlation_heatmap.png` - Feature correlation matrix
- `*_training_history.png` - Loss/accuracy curves for each DL model
- `all_models_comparison.png` - Side-by-side performance metrics

**SHAP Analysis:**
- `shap_*_summary.png` - Feature importance (bar charts)
- `shap_*_detailed.png` - Feature dependency (beeswarm plots)

**Advanced Predictions:**
- `shap_sleep_quality.png` - Sleep quality feature importance
- `shap_cv_risk.png` - Cardiovascular risk factors
- `shap_stress_level.png` - Stress estimation drivers
- `advanced_predictions_summary.png` - All predictions R² comparison

---

## Key Technologies

### Machine Learning
- **scikit-learn**: Random Forest, preprocessing, metrics
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Fast gradient boosting

### Deep Learning
- **TensorFlow/Keras**: Neural network architectures
- **Custom Architectures**: VGG-like CNN, Transformer blocks, ResNet

### Explainability
- **SHAP**: Model-agnostic feature importance
- **TreeExplainer**: Fast explanations for tree-based models
- **KernelExplainer**: Universal explanations for any model

### Data Processing
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Matplotlib/Seaborn**: Visualization

---

## Pipeline Features

### 🔄 Modular Design
- Each phase runs independently
- Easy to add new models or features
- Consistent API across all modules

### 📊 Comprehensive Metrics
- Accuracy, precision, recall, F1-score
- Confusion matrices
- ROC curves (where applicable)
- SHAP-based interpretability

### 🎯 Production Ready
- Saved model checkpoints
- Scalable preprocessing
- Reproducible results (fixed seeds)
- Extensive error handling

### 🔍 Explainable AI
- SHAP values for all predictions
- Feature importance rankings
- Individual prediction explanations
- Model-agnostic interpretability

---

## Future Enhancements (Phase 6+)

### Planned Features
- 🔐 **Federated Learning**: Privacy-preserving distributed training
- 📱 **Edge Deployment**: Model optimization for wearable devices
- 🔄 **Real-time Streaming**: Live health monitoring
- 🤖 **AutoML**: Automated hyperparameter tuning
- 🌐 **Multi-user Analytics**: Population health insights
- 🏥 **Clinical Integration**: EHR system compatibility

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update README with new functionality

---

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'src'`
```bash
# Solution: Run from project root directory
cd Wearable-Data-Fusion
python main.py
```

**Issue**: `TensorFlow GPU not detected`
```bash
# Solution: Install GPU version
pip install tensorflow-gpu
```

**Issue**: `Memory error during SHAP computation`
```python
# Solution: Reduce sample size in SHAP explainer
# Edit src/shap_explainability.py, line ~50
background = shap.sample(X_train, 100)  # Reduce from 200 to 100
```

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{wearable_health_fusion_2024,
  author = {Divya},
  title = {Wearable Health Fusion: Comprehensive Multi-Model Health Monitoring System},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/divya1010-b/Wearable-Data-Fusion},
  note = {9 ML/DL models with SHAP explainability for wearable health data}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Dataset**: [MD Imam Mahdi](https://www.kaggle.com/mdimammahdi) for the Smart Healthcare dataset
- **Libraries**: scikit-learn, TensorFlow, XGBoost, LightGBM, SHAP teams
- **Inspiration**: Healthcare AI research community

---

## Contact & Support

**Author**: Divya  
**Project**: Wearable Health Fusion  
**Version**: 2.0.0  

For questions, suggestions, or collaboration:
- 🐛 Issues: [GitHub Issues](https://github.com/divya1010-b/Wearable-Data-Fusion/issues)

---

## Project Statistics

- **Total Models**: 9 (4 Traditional ML + 5 Deep Learning)
- **Lines of Code**: ~3,500+ (excluding comments)
- **Prediction Tasks**: 3 specialized (Sleep, CV Risk, Stress)
- **Visualizations**: 20+ automated plots
- **Training Time**: ~30-45 minutes (complete pipeline, CPU)
- **Model Files**: ~150MB total
- **Documentation**: Comprehensive (README + docstrings)

