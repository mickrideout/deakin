# %% [markdown]
# Heart Disease Prediction Model Implementation
# This notebook implements a machine learning experiment for heart disease prediction

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import math
from autogluon.tabular import TabularPredictor
import random
import json
import os

# %% [markdown]
# Phase 1: Dataset Acquisition and Setup

# %%
df = pd.read_csv('framingham.csv')

print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())


# %%
# Handle missing values
print("\nMissing values before preprocessing:")
print(df.isnull().sum())


df['glucose'] = df['glucose'].fillna(df['glucose'].mode()[0])

# Impute missing values for numerical columns with median
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_columns:
    df[col] = df[col].fillna(df[col].median())

# Impute missing values for categorical columns with mode
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after imputation:")
print(df.isnull().sum())

df.drop(columns=['currentSmoker'], inplace=True)

# %%
plt.figure(figsize=(20, 15))
for i, column in enumerate(df.columns, 1):
    plt.subplot(4, 4, i)
    sns.histplot(data=df, x=column, hue='TenYearCHD', multiple="stack")
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Create bins for cigsPerDay based on custom ranges
df['cigsPerDay_binned'] = pd.cut(df['cigsPerDay'], 
                                bins=[-float('inf'), 0, 10, 20, 30, float('inf')],
                                labels=[0, 1, 2, 3, 4])
print("\nCigarettes per day distribution after binning:")
print(df['cigsPerDay_binned'].value_counts())
df.drop(columns=['cigsPerDay'], inplace=True)

# Apply log transformation to right-skewed numerical columns
skewed_columns = ['totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

for col in skewed_columns:
    df[col] = np.log1p(df[col])

df["glucose"] = np.log(np.log(df["glucose"])) # it is highly skewed

print("\nSkewness after log transformation:")
print(df[skewed_columns].skew())



# %%
# Implement Random Forest feature selection
features = df.drop('TenYearCHD', axis=1)
target = df['TenYearCHD']


# %%
# Data standardisation


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled = pd.DataFrame(features_scaled, columns=features.columns)

# %%
# Class imbalance handling using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(features_scaled, target)

print("\nClass distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# %%
plt.figure(figsize=(20, 15))
for i, column in enumerate(df.columns, 1):
    plt.subplot(4, 4, i)
    sns.histplot(data=df, x=column, hue='TenYearCHD', multiple="stack")
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
# Data splitting
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# AutoGluon model implementation
model_path = 'autogluon_heart_disease_model'
train_data = pd.concat([X_train, pd.Series(y_train, name='TenYearCHD')], axis=1)

if os.path.exists(model_path):
    print("Loading existing AutoGluon model...")
    predictor = TabularPredictor.load(model_path)
else:
    print("Training new AutoGluon model...")
    predictor = TabularPredictor(
        label='TenYearCHD',
        eval_metric='f1',
        path=model_path
    )
    predictor.fit(
        train_data,
        presets='best_quality',
        time_limit=600,
        num_bag_folds=5,
        num_stack_levels=2
    )
    print("Model saved to:", model_path)

# Make predictions and evaluate
y_pred = predictor.predict(X_test)
y_pred_proba = predictor.predict_proba(X_test)[1]

# %%

paper_results = {
    'Accuracy': 94.14,
    'Precision': 94.25,
    'Recall': 94.06,
    'F1-score': 94.06,
    'AUC-ROC': 99
}

accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
precision = round(precision_score(y_test, y_pred) * 100, 2)
recall = round(recall_score(y_test, y_pred) * 100, 2)
f1 = round(f1_score(y_test, y_pred) * 100, 2)
roc_auc = round(roc_auc_score(y_test, y_pred_proba) * 100, 2)

alternate_solution_results = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-score': f1,
    'AUC-ROC': roc_auc
}

print("\nAutoGluon Model Performance:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC:", roc_auc)


# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - AutoGluon Model')
plt.show()

# %%
print("\nAutoGluon Model Leaderboard:")
leaderboard = predictor.leaderboard()
print(leaderboard)

# %%
test_data = pd.concat([X_test, pd.Series(y_test, name='TenYearCHD')], axis=1)
feature_importance = predictor.feature_importance(data=test_data)
plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='index', data=feature_importance.reset_index())
plt.title('Feature Importance - AutoGluon Model')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print("\nFeature Importance Scores:")
print(feature_importance)

# %%
print("\nComparing Results:")
print("\nPaper Results:")
print(paper_results)

print("\nAlternate Solution Results:")
print(alternate_solution_results)

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Metric': list(paper_results.keys()),
    'Paper Results': list(paper_results.values()),
    'Alternate Solution': list(alternate_solution_results.values())
})

comparison_df.to_csv('alternative_solution_comparison_results.csv', index=False)


# Plot comparison
plt.figure(figsize=(10, 6))
comparison_df_melted = pd.melt(comparison_df, id_vars=['Metric'], 
                              value_vars=['Paper Results', 'Alternate Solution'],
                              var_name='Method', value_name='Score')
sns.barplot(x='Metric', y='Score', hue='Method', data=comparison_df_melted)
plt.title('Performance Comparison: Paper vs Alternate Solution')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# %%
