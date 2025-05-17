# Data Science Model Comparison Task Plan

## Overview
This task involves creating a comprehensive machine learning pipeline to compare various models for predicting a target variable in a network traffic dataset.

## Dataset Information
- File: Dataset4.csv
- Size: 123,117 entries
- Features: 84 columns (80 float64, 1 int64, 3 object)
- Target: 'target' column (categorical)

## Implementation Plan

### 1. Data Preprocessing
- Load Dataset4.csv into pandas DataFrame
- Handle missing values in target column
- Perform one-hot encoding for categorical features
- Scale numerical features using StandardScaler
- Apply SMOTE for class imbalance
- Feature selection (top 20 features)
- Target encoding using LabelEncoder
- Train/test split (80/20)

### 2. Task Set 1: Basic Model Implementation
- Implement and evaluate:
  - ResNet
  - GPlearn
  - Logistic Regression
- Compare performance using AUC and F1 score
- Perform hyperparameter optimisation

### 3. Task Set 2: Feature Importance Analysis
- Implement multiple feature importance methods
- Compare and interpret results
- Document insights and recommendations

### 4. Task Set 3: Ensemble Methods
- Implement and evaluate:
  - AutoGluon
  - LightGBM
- Compare with individual models
- Document deployment recommendations

### 5. Task Set 4: SVM Analysis
- Implement SVM for multiclass classification
- Evaluate performance
- Compare with other models

## Evaluation Metrics
- AUC Score
- F1 Score

## Visualisations
- Target distribution plot
- Correlation matrix
- Top feature correlations with target
- Model comparison visualisations

## Code Structure
- Jupyter-style annotations (# %%)
- Markdown sections (# %% [markdown])
- Minimal comments
- British spelling throughout 