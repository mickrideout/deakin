# %% [markdown]
# # Heart Disease Prediction Model
# In this notebook, we attempt to replicate the results of the paper "An efficient stacking-based ensemble technique for early heart attack prediction". We then implement an improved solution in an attempt to increase the accuracy of the model.

# %% [markdown]
# # Part 1 - Academic Paper Reproduction
# In this part, we attempt to replicate the results of the paper "An efficient stacking-based ensemble technique for early heart attack prediction". We do this in two sections, data preprocessing and model architecture.
# It is a fundamental tennant of science that findings should be reproducible. Difficulties in reproducing the paper's results will be discussed.

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## Data Preprocessing
# This sections loads the dataset and performs data clearning, transformation and feature selection all with the aim of reproducing the paper's results.

# %% [markdown]
# ### DataSet Loading


# %%
df = pd.read_csv('framingham.csv')

print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())



# %% [markdown]
# ### Missing Values
# Here we replicate the paper's approach to handling missing values. We perform explicity imputation as stated in the paper,
# and also impute the missing values for numerical columns with median and categorical columns with mode.
# Non feature selected columns are to be dropped anyhow so the superflous imputations for them is not a problem.

# %%
print("\nMissing values before preprocessing:")
print(df.isnull().sum())


# %% [markdown]
# Glucose was explicitly mentioned a having mode of values.

# %%
df['glucose'] = df['glucose'].fillna(df['glucose'].mode()[0])

# %% [markdown]
# Although not stated, impute the missing values for numerical columns with median and categorical columns with mode.
# The logics is that columns not used will be dropped anyhow

# %%
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_columns:
    df[col] = df[col].fillna(df[col].median())
    
print("Numeric columns which had median imputation:")
print(df.select_dtypes(include=['float64', 'int64']).columns)

print("\nMissing values after imputation:")
print(df.isnull().sum())

#%% [markdown]
# Takeaways are to alway explicity state what the imputations are and to what columns they are applied.

# %% [markdown]
# ### Outlier Removal
# Here we replicate the paper's approach to outlier removal. The columns totChol and sysBP are mentioned to have outliers.
# We remove the outliers for these columns using the IQR (Interquartile Range) method.

# %%
# Outlier removal for totChol and sysBP
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df = remove_outliers(df, 'totChol')
df = remove_outliers(df, 'sysBP')

print("\nDataset shape after outlier removal:", df.shape)

# %% [markdown]
# The number of rows removed wasnt stated in the paper. This would have been useful to verify if the outlier removal was equivalent

# %% [markdown]
# ### Data Standardisation
# Here we replicate the paper's approach to data standardisation. We use the StandardScaler from sklearn.

# %%

features = df.drop('TenYearCHD', axis=1)
target = df['TenYearCHD']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled = pd.DataFrame(features_scaled, columns=features.columns)

# %% [markdown]
# The paper provided the formula for standardisation which was useful.

# %% [markdown]
# ### Class Imbalance Handling
# Here we replicate the paper's approach to class imbalance handling. We use the SMOTE (Synthetic Minority Over-sampling Technique) from imblearn.
# Default SMOTE rebalancing produced more records than the paper's approach. So balanced resampling was needed. Default values were
# used as there was no information on the parameters used by the authors.
# Random seed of 42 is used to ensure reproducibility.


# %%
# Class imbalance handling using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(features_scaled, target)

print("\nClass distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Resample to get exactly 1697 records for each class
class_0 = X_resampled[y_resampled == 0]
class_1 = X_resampled[y_resampled == 1]

# Randomly sample 1697 records from each class
X_balanced = pd.concat([
    class_0.sample(n=1697, random_state=42),
    class_1.sample(n=1697, random_state=42)
])

y_balanced = pd.Series([0] * 1697 + [1] * 1697)

print("\nClass distribution after resampling:")
print(pd.Series(y_balanced).value_counts())

# %% [markdown]
# It is important to specify inputs and outputs from data preprocessing steps and the arguments used for algorithms. 
# Resampling of the smote output allowed the number of rows to be reduced to 3394, which is the same as the paper's approach.
# The random_state for the SMOTE rebalancer would also have been useful to specify.



# %% [markdown]
# ### Feature Correlation Analysis
# Here we replicate the paper's approach to feature correlation analysis. We use the correlation matrix from sklearn.
# We remove the highly correlated features using the correlation threshold of 0.85.

# %%
# Feature correlation analysis
correlation_matrix = X_resampled.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Remove highly correlated features (correlation > 0.85)
high_corr_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.85:
            high_corr_features.add(correlation_matrix.columns[i])

X_resampled = X_resampled.drop(columns=list(high_corr_features))
print("\nFeatures removed due to high correlation:", high_corr_features)
print("Remaining features:", X_resampled.columns.tolist())

# %% [markdown]
# The results from this step were the same as the paper's. No features were removed.

# %% [markdown]
# ### Feature Selection
# Here we replicate the paper's approach to feature selection. We use the RandomForestClassifier from sklearn.
# We select the top 10 features using the feature importance scores. 
# The paper alluded to the use of random forest for feature selection and to the number of features selected (10)
# Random seed of 42 is used to ensure reproducibility.


# %%
# Implement Random Forest feature selection
rf_selector = RandomForestClassifier(random_state=42)
rf_selector.fit(features, target)

# Get feature importance scores
feature_importance = pd.DataFrame({
    'feature': features.columns,
    'importance': rf_selector.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Select top 10 features
top_10_features = feature_importance.head(10)['feature'].tolist()
# X_selected = X_resampled[top_10_features]

print("Top 10 selected features:")
print(top_10_features)

# The assumed features selected by the paper. The 10 ten features in Figure 2 Visualize the features score of random forest features
paper_features = ["age", "sysBP", "BMI", "totChol", "diaBP", "glucose", "heartRate", "cigsPerDay", "education", "prevalentHyp"]


X_selected = X_resampled[paper_features]

# %% [markdown]
# The paper did not state explicity what the features were. So we had to make an assumption that it was the top 10 features 
# based on the figure 2. Based on this, the features determined as important were not the same as the paper's, even though
# the same method was used (RF). The male column was not included in the paper's features. prevalentHyp was included in the paper's
# features and not in ours. Again the paper lacked information to replicate feature selection results. The random seed was not specified
# so the results were not reproducible.

# %% [markdown]
# ### Train / Test Split
# The train / test split ratio was stated in the paper as 70 / 30. However the strategy was not specified.
# We used random_state=42 to ensure reproducibility for our random splitter.


# %%
# Data splitting
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_resampled, test_size=0.3, random_state=42
)

# %% [markdown]
# The strategy was not specified in the paper. If a random splitter was used, the results would not be reproducible.


# %%
# ### Model Architecture and Evaluation
# Here we train and evaluate the models as described in the paper. We use the XGBoost, Random Forest, Decision Tree and KNN models.
# We also use the FT-DNN and DNN models. We use the same architecture as the paper's.
# The models are evaludated using the accuracy, precision, recall, f1-score and roc-auc score.
# Any discrepanncy in the results will be discussed in the results section.


# %%
# The model results from the paper
paper_results = pd.DataFrame({
    'Model': ['RF', 'KNN', 'DT', 'XGB', 'FT-DNN', 'DNN', 'ML_Ensemble', 'MDLSM'],
    'Accuracy': [94.02, 93.45, 92.35, 94.03, 80.19, 76.73, 94.1, 94.14],
    'Precision': [94.01, 93.53, 92.23, 94.03, 77.03, 72.85, 94.04, 94.25],
    'Recall': [94.01, 93.21, 92.22, 94.02, 86.77, 86.19, 94.05, 94.06],
    'F1-score': [94.01, 93.25, 91.22, 94.02, 69.43, 67.32, 94.05, 94.06],
    'AUC-ROC': [98, 93, 91, 98, 86.2, 83.1, 99, 99]
})
paper_results.set_index('Model', inplace=True)



# %%
# Function to train and evaluate the models
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    results = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred)
    }
    
    if y_pred_proba is not None:
        results['AUC-ROC'] = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{model_name} Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    return model, results

# %% [markdown]
# ### Baseline ML Models
# Here we train the baseline ML models as described in the paper. We use the XGBoost, Random Forest, Decision Tree and KNN models.
# The models are evaludated using the accuracy, precision, recall, f1-score and roc-auc score.
# Default values were used for the models as there was no information on the parameters used by the authors.
# Random seed of 42 is used to ensure reproducibility.

# 
models = {
    'XGB': XGBClassifier(random_state=42, tree_method='hist'),
    'RF': RandomForestClassifier(random_state=42),
    'DT': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

baseline_results = {}
for name, model in models.items():
    trained_model, results = train_and_evaluate_model(
        model, X_train, X_test, y_train, y_test, name
    )
    baseline_results[name] = results

# %% [markdown]
# All models listed in the paper should have specified the hyperparameters used.



# %% [markdown]
# ### Deep Learning Models
# Here we implement the deep learning models from the paper. Those models are FT-DNN and DNN.


# %% [markdown]
# ### FT-DNN Architecture
# The FT-DNN architecture was described a follows:
# A 4 layer feedforward neural network with 16, 12, 8 and 4 neurons in the layers respectively.
# ReLU is the activation function.
# Adam was the optimizer.
# Binary crossentropy was the loss function.
# Learning rate of 0.001 was specified.
# The metrics were not specified. So we used the f1_score metric.

# %%
def create_ft_dnn():
    model = Sequential([
        Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(12, activation='relu'),
        Dense(8, activation='relu'),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['f1_score']
    )
    return model



# %% [markdown]
# ### DNN Architecture
# The DNN architecture was described a follows:
# A 4 layer feedforward neural network with 12, 10, 8 and 6 neurons in the layers respectively.
# ReLU is the activation function.
# Adam was the optimizer.
# Binary crossentropy was the loss function.
# The metrics were not specified. So we used the f1_score metric.
# No learning rate was specified so we did not specify one either.

# %%
def create_dnn():
    model = Sequential([
        Dense(12, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(10, activation='relu'),
        Dense(8, activation='relu'),
        Dense(6, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy',
        metrics=['f1_score']
    )
    return model

# %% [markdown]
# ### Deep Learning Model Training
# We have had to use common sense values for the training hyperparameters as none were specified.
# We will use epochs=400, batch_size=32, validation_split=0.1, verbose=0 as the default values.

# %%
# Train DL models
ft_dnn = create_ft_dnn()
dnn = create_dnn()

# Train FT-DNN
ft_dnn_history = ft_dnn.fit(
    X_train, y_train,
    epochs=400,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)

# Train DNN
dnn_history = dnn.fit(
    X_train, y_train,
    epochs=400,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)

# Evaluate DL models
dl_models = {
    'FT-DNN': ft_dnn,
    'DNN': dnn
}

dl_results = {}
for name, model in dl_models.items():
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_pred_proba = model.predict(X_test)
    
    results = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n{name} Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    dl_results[name] = results

# %% [markdown]
# The paper did not specify the training hyperparameters for the deep learning models. This is crucial information
# for reproducibility.

# %%
# ### Ensemble Models
# Here we implement the ensemble models as described in the paper. We use the ML Ensemble and MDLSM models.

# %% [markdown]
# ## ML Ensemble Model
# The ML Ensemble model is a simple ensemble of the baseline ML models.
# It aggregates the predictions of the baseline ML models using a simple mean vote.

# %%
class MLEnsemble:
    def __init__(self, models):
        self.models = models
    
    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0) > 0.5
    
    def predict_proba(self, X):
        predictions = np.array([model.predict_proba(X)[:, 1] for model in self.models])
        return np.mean(predictions, axis=0)

# %% [markdown]
# ## MDLSM Model
# The MDLSM model is a meta-learning model that uses the ML Ensemble model to predict the class labels and the deep learning models to predict the probabilities.
# It aggregates the predictions of the ML Ensemble model and the deep learning models using a simple mean vote.

# %%
class MDLSM:
    def __init__(self, ml_ensemble, ft_dnn, dnn):
        self.ml_ensemble = ml_ensemble
        self.ft_dnn = ft_dnn
        self.dnn = dnn
    
    def predict(self, X):
        ml_pred = self.ml_ensemble.predict(X)
        ft_dnn_pred = (self.ft_dnn.predict(X).flatten() > 0.5).astype(int)
        dnn_pred = (self.dnn.predict(X).flatten() > 0.5).astype(int)
        
        predictions = np.vstack([ml_pred, ft_dnn_pred, dnn_pred])
        return np.mean(predictions, axis=0) > 0.5
    
    def predict_proba(self, X):
        ml_proba = self.ml_ensemble.predict_proba(X)
        ft_dnn_proba = self.ft_dnn.predict(X).flatten()
        dnn_proba = self.dnn.predict(X).flatten()
        
        predictions = np.vstack([ml_proba, ft_dnn_proba, dnn_proba])
        return np.mean(predictions, axis=0)

# Train ensemble models
ml_ensemble = MLEnsemble([
    models['DT'],
    models['XGB'],
    models['KNN'],
    models['RF']
])
ml_ensemble.fit(X_train, y_train)

mdlsm = MDLSM(ml_ensemble, ft_dnn, dnn)

# Evaluate ensemble models
ensemble_models = {
    'ML Ensemble': ml_ensemble,
    'MDLSM': mdlsm
}

ensemble_results = {}
for name, model in ensemble_models.items():
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    results = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred)
    }
    
    if y_pred_proba is not None:
        results['AUC-ROC'] = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{name} Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    ensemble_results[name] = results 

# %%
# Combine all models for evaluation
all_models = {
    **models,  # Baseline ML models
    **dl_models,  # Deep Learning models
    **ensemble_models  # Ensemble models
}

# %% [markdown]
# ### Model Evaluation and Result Verification
# Here we evaluate the models and compare the results with the paper's results.

# %%
# Combine all models for evaluation
# Combine all results
all_results = {
    **baseline_results,
    **dl_results,
    **ensemble_results
}

# Create comparison DataFrame
comparison_df = pd.DataFrame(all_results).T
print("\nModel Performance Comparison:")
print(comparison_df)


# %%
# Save results to CSV
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC']

for metric in metrics:
    comparison_df[metric] = (comparison_df[metric] * 100).round(2)

# %%



print("\nComparing our results with paper results:")
comparison = pd.DataFrame({
    'Our Results': comparison_df[metrics].values.flatten(),
    'Paper Results': paper_results[metrics].values.flatten()
}, index=pd.MultiIndex.from_product([comparison_df.index, metrics], names=['Model', 'Metric']))
comparison['Difference (%)'] = ((comparison['Our Results'] - comparison['Paper Results']) / comparison['Paper Results'] * 100).round(2)
print(comparison)

comparison.to_csv('model_comparison_results.csv')
print("\nResults have been saved to 'model_comparison_results.csv'")

plt.figure(figsize=(15, 10))
x = np.arange(len(comparison_df.index))
width = 0.35

for i, metric in enumerate(metrics):
    plt.subplot(2, 3, i+1)
    plt.bar(x - width/2, comparison_df[metric], width, label='Replication Results')
    plt.bar(x + width/2, paper_results[metric], width, label='Paper Results')
    
    plt.xlabel('Models')
    plt.ylabel(f'{metric.capitalize()} (%)')
    plt.title(f'Comparison of {metric.capitalize()}')
    plt.xticks(x, comparison_df.index, rotation=45)
    plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Model Comparison
# We were not able to replicate the results of the paper exactly as there were some differences in the results.
# The MDSLM model (the final ensemble model) had replication results for f1-score being 83.34 which was 11.4% lower than the paper results of 94.06
# Most of the replication results show lower performance than the paper results.
# Some variations, such as higher f1-scores for DNN and FT-DNN and higher AUC-ROC for RF in the replication were observed.



# %% [markdown]
# ## Replication Conclusion
# Whilst it was possible to replicate the architecture of the models, the results were not able to be replicated exactly.
# The inability to replicate the results accurately is mostly attributable to the 
# lack of crucial information in the paper. The main items of information that were missing were:
# - The hyperparameters used for the models
# - The training hyperparameters for the deep learning models.
# - The train / test split ratio and methodology.
# - Exact row counts for steps in the data cleaning process.
# - Library versions used for the models.
# - Random seeds used for the models.
# - More precise information on the data preprocessing steps used.
# 



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


# %%
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
        num_stack_levels=2,
        random_seed=42
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



