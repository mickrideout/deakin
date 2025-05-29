# %% [markdown]
# # Part 2 - Heart Disease Prediction Alternate Solution
# This notebook implements an alternate solution to the heart disease prediction problem. 
# It uses AutoGluon to train a model and evaluate its performance.

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
# ### Dataset Loading
# We reload the dataset to clearn any modifications made in part 1

# %%
df = pd.read_csv('framingham.csv')

print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# %% [markdown]
# ## Data Preprocessing
# This sections loads the dataset and performs data clearning and transformation

# %% [markdown]
# ### Missing Value Imputation
# Here we impute all numerical columns with median if they have missing values.

# %%
print("\nMissing values before preprocessing:")
print(df.isnull().sum())

# Impute missing values for numerical columns with median
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_columns:
    df[col] = df[col].fillna(df[col].median())

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

# %% [markdown]
# ### Skewness Transformation
# Here we deal with the right skewness of some numerical columns. Binning is also performed on cigsPerDay.


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

# %% [markdown]
# After these transforms, the features have a normal distribution and the cigsPerDay column is binned with 5 bins.

# %%
features = df.drop('TenYearCHD', axis=1)
target = df['TenYearCHD']


# %% [markdown]
# ### Data Standardisation
# Here we standardise the data.


# %%
# Data standardisation


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled = pd.DataFrame(features_scaled, columns=features.columns)

# %% [markdown]
# ### Class Imbalance Handling
# Here we handle the class imbalance using SMOTE as the target variable is imbalanced.


# %%
# Class imbalance handling using SMOTE
print("\nClass distribution before SMOTE:")
print(pd.Series(target).value_counts())


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(features_scaled, target)

print("\nClass distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# %% [markdown]
# Plot the features histograms to show the affects of transformations.

# %%
plt.figure(figsize=(20, 15))
for i, column in enumerate(df.columns, 1):
    plt.subplot(4, 4, i)
    sns.histplot(data=df, x=column, hue='TenYearCHD', multiple="stack")
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Data Splitting
# Here we split the data into training and testing sets with a 80-20 split using a random splitting method

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# %% [markdown]
# ### AutoGluon Model Implementation
# Here we implement the AutoGluon model, if a model doesnt already exist on disk. After training, the model is saved to disk.


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

# %% [markdown]
# ### Model Evaluation
# Here we evaluate the model's performance.


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

# %% [markdown]
# ### AutoGluon Model Leaderboard
# Show the top performing models generated by AutoGluon.

# %%
print("\nAutoGluon Model Leaderboard:")
leaderboard = predictor.leaderboard()
print(leaderboard)

print("\nBest Model Information:")
best_model_name = leaderboard.iloc[0]['model']
print(f"Best Model: {best_model_name}")
print("\nModel Hyperparameters:")
print(predictor.model_info(best_model_name))


# %% [markdown]
# ### AutoGluon Feature Importance
# Here we plot the feature importance of the best model.

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

# %% [markdown]
# The order of importance of the AutoGluon model is different to the paper's feature importance rank order. Smoking and heart readings 
# had more importance in the AutoGluon model.

# %%

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Metric': list(paper_results.keys()),
    'Paper Results': list(paper_results.values()),
    'Alternate Solution': list(alternate_solution_results.values())
})

comparison_df.to_csv('alternative_solution_comparison_results.csv', index=False)

print("\nComparison Results Table:")
print(comparison_df.to_string(index=False))



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

# %% [markdown]
# ### Model Comparison
# The model performance of our alternate solution was across the board on all metrics approximately 3% better than the paper's results.
# This is a good result as it shows that AutoGluon can be used to train a model that performs well on the heart disease prediction problem.
# On top of this, our model was able to output the feature importance of the best model which aids in the interpretability of the model.
# The paper's implementation did not allow for this type of output


# %% [markdown]
# ## Conclusion
# The AutoGluon model was able to produce a more performant model than the paper's implementation.
# In addition, the AutoGluon model was able to output the feature importance of the best model which aids in the interpretability of the model.
# Improved data preprocessing was also an enabling factor in the performance of the AutoGluon model.

