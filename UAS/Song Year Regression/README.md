# End-to-End Regression: Song Release Year Prediction

## 📋 Project Overview

This project implements a comprehensive regression pipeline to predict song release years from audio features. The solution compares multiple machine learning algorithms and deep learning approaches to find the best model for this regression task.

## 🎯 Main Objective

Design and implement an end-to-end regression pipeline (using machine learning and deep learning) that can accurately predict the release year of a song from its audio characteristics.

## 👤 Student Information

- **Name**: Devon Tamaam Adira S
- **Class**: [Your Class]
- **NIM**: [Your Student ID]
- **Course**: Machine Learning - Telkom University
- **Assignment**: UAS (Final Exam) - Individual Task

## 📊 Dataset Description

### Dataset: `midterm-regresi-dataset.csv`

- **Structure**: Each row represents one song
- **Target Variable**: First column - Release year (e.g., 2001, 1995)
- **Features**: Remaining columns (90 features) - Audio characteristics
  - Timbre features
  - Acoustic properties
  - Other music signal characteristics
  - Feature names: feature_1, feature_2, ..., feature_90

The audio features are numeric values computed from the music signal that describe various aspects of the sound, though they don't have simple human-friendly interpretations.

## 🔧 Technical Implementation

### 1. Data Preprocessing
- **Feature Scaling**: StandardScaler for normalization
- **Train-Test Split**: 80-20 split for model validation
- **Outlier Detection**: IQR method (keeping all valid years)
- **Missing Value Check**: Ensuring data integrity

### 2. Regression Models Implemented

#### Traditional Machine Learning:
1. **Linear Regression**
   - Baseline model
   - Simple linear relationships
   - Fast and interpretable

2. **Ridge Regression (L2)**
   - Regularization to prevent overfitting
   - Better for correlated features
   - Alpha = 1.0

3. **Lasso Regression (L1)**
   - Feature selection capability
   - Sparse solution
   - Alpha = 0.1

4. **Random Forest Regressor**
   - Ensemble of decision trees
   - Feature importance analysis
   - 100 estimators, max_depth=15

5. **XGBoost Regressor**
   - Gradient boosting framework
   - High performance
   - Built-in regularization

6. **LightGBM Regressor**
   - Fast gradient boosting
   - Efficient for large datasets
   - Lower memory usage

#### Deep Learning:
7. **Neural Network**
   - Architecture:
     - Input Layer → Dense(256) → BatchNorm → Dropout(0.3)
     - Dense(128) → BatchNorm → Dropout(0.3)
     - Dense(64) → BatchNorm → Dropout(0.2)
     - Dense(32) → Dropout(0.2)
     - Output Layer: Dense(1)
   - Optimizer: Adam (lr=0.001)
   - Loss: Mean Squared Error (MSE)
   - Callbacks: EarlyStopping, ReduceLROnPlateau

### 3. Evaluation Metrics

All models are evaluated using:
- **MSE (Mean Squared Error)**: Average squared difference
- **RMSE (Root Mean Squared Error)**: Square root of MSE (in years)
- **MAE (Mean Absolute Error)**: Average absolute difference (in years)
- **R² Score**: Coefficient of determination (proportion of variance explained)

**Primary Metric**: R² Score (higher is better, 1.0 = perfect prediction)

## 📁 Project Structure

```
EndToEnd-Regression/
│
├── midterm-regresi-dataset.csv            # Dataset with audio features
├── Song_Year_Prediction_Regression.ipynb  # Main notebook with pipeline
└── README.md                              # This file
```

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm tensorflow scipy
```

### Execution Steps

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook Song_Year_Prediction_Regression.ipynb
   ```

2. **Run All Cells**: Execute cells sequentially or use "Run All"

3. **Pipeline Stages**:
   - Data loading and exploration
   - Target distribution analysis
   - Feature preprocessing and scaling
   - Train-test split
   - Training 7 different models
   - Model evaluation and comparison
   - Prediction analysis and visualization
   - Feature importance analysis

4. **Output**: Comprehensive comparison of all models with visualizations

## 📈 Model Results

The notebook provides detailed comparison across all metrics:

| Model | Train RMSE | Test RMSE | Train MAE | Test MAE | Train R² | Test R² |
|-------|-----------|----------|-----------|----------|----------|---------|
| Linear Regression | [Run to see] | [Run to see] | [Run to see] | [Run to see] | [Run to see] | [Run to see] |
| Ridge Regression | [Run to see] | [Run to see] | [Run to see] | [Run to see] | [Run to see] | [Run to see] |
| Lasso Regression | [Run to see] | [Run to see] | [Run to see] | [Run to see] | [Run to see] | [Run to see] |
| Random Forest | [Run to see] | [Run to see] | [Run to see] | [Run to see] | [Run to see] | [Run to see] |
| XGBoost | [Run to see] | [Run to see] | [Run to see] | [Run to see] | [Run to see] | [Run to see] |
| LightGBM | [Run to see] | [Run to see] | [Run to see] | [Run to see] | [Run to see] | [Run to see] |
| Neural Network | [Run to see] | [Run to see] | [Run to see] | [Run to see] | [Run to see] | [Run to see] |

*Run the notebook to populate with actual results*

### Best Model Selection
The best model is automatically selected based on **Test R² score**, which measures how well the model explains variance in the test data.

## 📊 Visualizations Included

1. **Target Distribution**: Histogram, box plot, and violin plot of release years
2. **Feature Distributions**: Sample of first 12 audio features
3. **Training History**: Loss, MAE, and RMSE curves (Neural Network)
4. **Model Comparison**: Bar charts for RMSE, MAE, and R² across all models
5. **Actual vs Predicted**: Scatter plots for each model
6. **Residual Analysis**: 
   - Residual scatter plot
   - Residual distribution histogram
   - Q-Q plot for normality check
7. **Feature Importance**: Top 20 most important features (Random Forest)

## 🔍 Key Insights

### Challenges Addressed:
1. **High Dimensionality**: 90 audio features
   - Solution: Feature scaling and regularization techniques
   
2. **Complex Relationships**: Non-linear patterns in audio-to-year mapping
   - Solution: Tree-based and neural network models
   
3. **Model Selection**: Finding optimal balance between bias and variance
   - Solution: Comprehensive comparison with multiple metrics

### Best Practices Implemented:
- ✅ Proper train-test split (80-20)
- ✅ Feature scaling for distance-based models
- ✅ Multiple model comparison
- ✅ Comprehensive evaluation metrics
- ✅ Residual analysis for model diagnostics
- ✅ Feature importance for interpretability
- ✅ Early stopping to prevent overfitting
- ✅ Visualization of predictions and errors

## 📚 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
tensorflow>=2.8.0
scipy>=1.7.0
```

## 🎓 Learning Outcomes

Through this project, the following skills were developed:
1. Complete regression pipeline from data to predictions
2. Handling high-dimensional numerical data
3. Implementation of multiple regression algorithms
4. Deep learning for regression tasks
5. Model evaluation with appropriate metrics
6. Residual analysis and diagnostics
7. Feature importance interpretation
8. Comparative analysis of model performance

## 🔮 Future Improvements

Potential enhancements:
1. **Hyperparameter Tuning**: Grid/Random search for optimal parameters
2. **Feature Engineering**: Create polynomial features, feature interactions
3. **Ensemble Methods**: Stacking or voting regressors
4. **Time-Series Analysis**: Consider temporal patterns in music evolution
5. **Advanced Neural Architectures**: LSTM or Transformer models
6. **Cross-Validation**: K-fold CV for more robust evaluation
7. **Feature Selection**: Recursive feature elimination

## 📖 References

- Scikit-learn Documentation: https://scikit-learn.org/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- TensorFlow/Keras Documentation: https://www.tensorflow.org/
- Regression Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

## 💡 Interpretation Guide

### Understanding the Metrics:
- **RMSE < 10 years**: Excellent performance
- **10 ≤ RMSE < 15 years**: Good performance
- **RMSE ≥ 15 years**: Room for improvement
- **R² > 0.7**: Strong predictive power
- **R² between 0.5-0.7**: Moderate predictive power
- **R² < 0.5**: Weak predictive power

### Typical Results:
For audio-based year prediction, an MAE of 8-10 years is considered good, as audio features have inherent ambiguity and music styles can span multiple years.

## 📝 Notes

- All code is fully documented with explanations
- Visualizations help understand model behavior
- Results are reproducible with fixed random seeds
- Models are saved and can be reused

---

**Repository**: [Telkom-University-Machine-Learning-Repo](https://github.com/Chroma321/Telkom-University-Machine-Learning-Repo)

**Last Updated**: January 4, 2026
