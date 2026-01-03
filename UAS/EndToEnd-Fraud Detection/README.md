# End-to-End Fraud Detection Machine Learning Project

## 📋 Project Overview

This project implements a comprehensive machine learning pipeline to predict the probability of online transactions being fraudulent. The solution includes both traditional machine learning algorithms and deep learning approaches, with extensive data preprocessing, feature engineering, and model evaluation.

## 🎯 Main Objective

Design and implement an end-to-end machine learning and deep learning pipeline that can predict the probability of an online transaction being fraudulent with high accuracy and reliability.

## 👤 Student Information

- **Name**: Devon Tamaam Adira S
- **Class**: [Your Class]
- **NIM**: [Your Student ID]
- **Course**: Machine Learning - Telkom University
- **Assignment**: UAS (Final Exam) - Individual Task

## 📊 Dataset Description

### Training Dataset (`train_transaction.csv`)
- Contains labeled transaction data used for model training and evaluation
- Each row represents one online transaction with multiple features:
  - Transaction amount
  - Time information
  - Product codes
  - Card information
  - Address details
  - Binary label `isFraud` (1 = fraudulent, 0 = legitimate)

### Test Dataset (`test_transaction.csv`)
- Contains unlabeled transaction data with the same feature columns (except `isFraud`)
- Used as input to generate fraud probability predictions
- Output format: `(TransactionID, isFraud probability)`

## 🔧 Technical Implementation

### 1. Data Preprocessing
- **Missing Value Handling**: Median imputation for numerical features, label encoding for categorical features
- **Feature Encoding**: Label encoding for categorical variables with handling of unseen categories
- **Feature Scaling**: StandardScaler for numerical features
- **Feature Selection**: Removal of features with >80% missing values

### 2. Class Imbalance Handling
- **Technique Used**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Purpose**: Balance the highly imbalanced fraud detection dataset
- **Impact**: Improved model sensitivity to fraud cases

### 3. Machine Learning Models

#### Traditional ML Models:
1. **Logistic Regression**
   - Baseline linear classifier
   - Fast training and inference
   - Interpretable coefficients

2. **Random Forest**
   - Ensemble of decision trees
   - Feature importance analysis
   - Robust to overfitting

3. **XGBoost**
   - Gradient boosting framework
   - High performance on structured data
   - Built-in regularization

4. **LightGBM**
   - Fast gradient boosting
   - Efficient memory usage
   - Excellent for large datasets

#### Deep Learning Model:
5. **Neural Network**
   - Architecture:
     - Input Layer → Dense(256) → BatchNorm → Dropout(0.3)
     - Dense(128) → BatchNorm → Dropout(0.3)
     - Dense(64) → BatchNorm → Dropout(0.2)
     - Dense(32) → Dropout(0.2)
     - Output Layer: Dense(1, sigmoid)
   - Optimizer: Adam (lr=0.001)
   - Loss: Binary Crossentropy
   - Callbacks: EarlyStopping, ReduceLROnPlateau

### 4. Evaluation Metrics

All models are evaluated using comprehensive metrics:
- **Accuracy**: Overall correctness
- **Precision**: Proportion of positive identifications that were correct
- **Recall**: Proportion of actual positives that were identified correctly
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve (primary metric)
- **Confusion Matrix**: Detailed breakdown of predictions

## 📁 Project Structure

```
EndToEnd-Fraud Detection/
│
├── train_transaction.csv              # Training dataset
├── test_transaction.csv               # Test dataset
├── Fraud_Detection_EndToEnd.ipynb     # Main notebook with complete pipeline
├── fraud_detection_submission.csv     # Generated predictions for test data
└── README.md                          # This file
```

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm tensorflow imbalanced-learn
```

### Execution Steps

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook Fraud_Detection_EndToEnd.ipynb
   ```

2. **Run All Cells**: Execute cells sequentially or use "Run All"

3. **Pipeline Stages**:
   - Data loading and exploration
   - Data preprocessing and feature engineering
   - Train-validation split
   - Class imbalance handling (SMOTE)
   - Training 5 different models
   - Model evaluation and comparison
   - Prediction generation on test data

4. **Output**: The notebook generates `fraud_detection_submission.csv` with fraud probabilities for test transactions

## 📈 Model Results

The notebook provides comprehensive model comparison across all metrics:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | [Run to see] | [Run to see] | [Run to see] | [Run to see] | [Run to see] |
| Random Forest | [Run to see] | [Run to see] | [Run to see] | [Run to see] | [Run to see] |
| XGBoost | [Run to see] | [Run to see] | [Run to see] | [Run to see] | [Run to see] |
| LightGBM | [Run to see] | [Run to see] | [Run to see] | [Run to see] | [Run to see] |
| Neural Network | [Run to see] | [Run to see] | [Run to see] | [Run to see] | [Run to see] |

*Run the notebook to populate the table with actual results*

### Best Model Selection
The best model is automatically selected based on **ROC-AUC score**, which is the most appropriate metric for imbalanced classification problems like fraud detection.

## 📊 Visualizations Included

1. **Class Distribution**: Bar and pie charts showing fraud vs legitimate transactions
2. **Missing Values**: Horizontal bar chart of features with missing data
3. **Training History**: Loss, accuracy, AUC, precision, and recall curves (Neural Network)
4. **Model Comparison**: Bar charts comparing all metrics across models
5. **ROC Curves**: Comparative ROC curves for all models
6. **Confusion Matrices**: Heatmaps for each model's predictions
7. **Feature Importance**: Top 20 most important features (Random Forest)

## 🔍 Key Insights

### Challenges Addressed:
1. **High Class Imbalance**: Fraud cases are typically <5% of all transactions
   - Solution: SMOTE oversampling technique
   
2. **Missing Data**: Many features have substantial missing values
   - Solution: Strategic imputation and feature selection
   
3. **Feature Complexity**: Hundreds of features with varying types
   - Solution: Systematic preprocessing pipeline

4. **Model Selection**: Finding the right balance between performance and interpretability
   - Solution: Compare multiple algorithms and select based on ROC-AUC

### Best Practices Implemented:
- ✅ Stratified train-validation split to preserve class distribution
- ✅ Separate preprocessing for train and test data (no data leakage)
- ✅ Handling of unseen categories in test data
- ✅ Comprehensive evaluation with multiple metrics
- ✅ Feature importance analysis for interpretability
- ✅ Early stopping and learning rate reduction for deep learning
- ✅ Proper handling of imbalanced data

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
imbalanced-learn>=0.9.0
```

## 🎓 Learning Outcomes

Through this project, the following skills were developed:
1. Complete ML pipeline development from raw data to predictions
2. Handling real-world data challenges (missing values, imbalance, etc.)
3. Implementation of multiple ML and DL algorithms
4. Model evaluation and comparison techniques
5. Feature engineering and selection strategies
6. Deep learning model architecture design
7. Proper validation and testing procedures

## 🔮 Future Improvements

Potential enhancements for this project:
1. **Hyperparameter Tuning**: Grid/Random search for optimal parameters
2. **Advanced Feature Engineering**: Create interaction features, time-based features
3. **Ensemble Methods**: Combine predictions from multiple models
4. **Anomaly Detection**: Implement unsupervised methods like Isolation Forest
5. **Model Interpretability**: Add SHAP values or LIME for explainability
6. **Production Deployment**: Create API for real-time fraud detection

## 📖 References

- Scikit-learn Documentation: https://scikit-learn.org/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- TensorFlow Documentation: https://www.tensorflow.org/
- SMOTE Paper: https://arxiv.org/abs/1106.1813

## 📝 Notes

- The notebook is fully documented with markdown explanations for each step
- All code includes comments for clarity
- Visualizations are provided for better understanding
- Results are reproducible with fixed random seeds

---

**Repository**: [Telkom-University-Machine-Learning-Repo](https://github.com/Chroma321/Telkom-University-Machine-Learning-Repo)

**Last Updated**: January 4, 2026
