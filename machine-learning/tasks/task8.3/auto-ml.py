# %% [markdown]
# # Network Attack Classification for Task 8.3
# 
# This notebook implements various models for predicting network traffic patterns.

# %%
import os
import time
import multiprocessing

import numpy as np
import pandas as pd
from scipy.stats import uniform, loguniform

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# %%
# Configuration
optimise_models = True  # Set to True to perform hyperparameter optimisation
n_jobs = -1  # Use all available cores
n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

# %%
df = pd.read_csv('Dataset4.csv')
print(f"Rows in dataset: {df.shape[0]}, Columns in dataset: {df.shape[1]}")

# %% [markdown]
# ## Data Preprocessing

# %% [markdown]
# Drop rows where target is 'na' or missing

# %%
df.dropna(subset=['target'], inplace=True)
df = df[df['target'] != 'na']
print(f"Rows in dataset after dropping missing values: {df.shape[0]}, Columns in dataset after dropping missing values: {df.shape[1]}")

# %% [markdown]
# There were only two rows with missing values.

# %% [markdown]
# Display the target class counts

# %%
# Count and display target classes
print("Number of classes:", df['target'].nunique())
print("\nValue counts for each target class:")
print(df['target'].value_counts())

# %% [markdown]
# Classes are imbalanced so this needs to be handled

# %% [markdown]
# Check for missing values

# %%
# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values[missing_values > 0])

# %% [markdown]
# MIssing values exist so need handling, fill with median for numerical columns

# %%
# Handle missing values by filling with median for numerical columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# %% [markdown]
# One-hot encoding for categorical columns (excluding target)

# %%
categorical_columns = df.select_dtypes(include=['object']).columns
categorical_columns = categorical_columns[categorical_columns != 'target']
df = pd.get_dummies(df, columns=categorical_columns)

# %% [markdown]
# Encode target variable as it is read as a string object

# %%
le = LabelEncoder()
df['target'] = le.fit_transform(df['target'])


# %% [markdown]
# Show top correlations with target

# %%
correlations = df.corr()['target'].sort_values(ascending=False)
print("\nTop 10 correlations with target:")
print(correlations.head(10))

# %% [markdown]
# Visualise target distribution to assess how bad the imbalance is

# %%
plt.figure(figsize=(10, 6))
sns.countplot(x='target', data=df)
plt.title('Distribution of Target Variable')
plt.show()

# %% [markdown]
# Data is highly imbalanced for class 2

# %% [markdown]
# Show correlation matrix heatmap

# %%
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# %% [markdown]
# Some features are highly correlated with each other, future investigation should handle this.

# %%
# Prepare features and target
X = df.drop('target', axis=1)
y = df['target'].astype(int)  # Ensure target is integer type

# %% [markdown]
# Scale features

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %% [markdown]
# Perform automated feature selection

# %%
selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = X.columns[selector.get_support()].tolist()
print("\nSelected features:", selected_features)

# %% [markdown]
# The top 20 features are selected and used for all subsequent models

# %% [markdown]
# Apply SMOTE to handle class imbalance

# %%
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_selected, y)

# %% [markdown]
# Resampling introduced a large number of samples which increased computational complexity. Reduce the size to 1/10th for each class.

# %%
print("\nOriginal resampled dataset shape:", X_resampled.shape)

resampled_df = pd.DataFrame(X_resampled)
resampled_df['target'] = y_resampled.astype(int)  # Ensure target is integer type

unique_classes = resampled_df['target'].unique()
reduced_df = pd.DataFrame(columns=resampled_df.columns)

# For each class, take 10% of the samples
for cls in unique_classes:
    class_df = resampled_df[resampled_df['target'] == cls]
    # Use stratified sampling to maintain distribution within each class
    reduced_class_df = class_df.sample(frac=0.1, random_state=42)
    reduced_df = pd.concat([reduced_df, reduced_class_df])

# Shuffle the reduced dataset
reduced_df = reduced_df.sample(frac=1, random_state=42).reset_index(drop=True)

X_resampled = reduced_df.drop('target', axis=1).values
y_resampled = reduced_df['target'].values.astype(int)  # Ensure target is integer type

print("Reduced resampled dataset shape:", X_resampled.shape)
print(f"Reduction: {100 * (1 - X_resampled.shape[0] / resampled_df.shape[0]):.2f}% of original size")

# %% [markdown]
# Split the dataset into training and test sets, no need for stratified sampling as the classes are balanced.

# %%
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# %% [markdown]
# ## Task 1: Multiple Supervised Learning Models
# 
# Implementing and evaluating ResNet, KNN, and Logistic Regression models. Evaluation is done using f1-score and AUC score.

# %% [markdown]
# Function to evaluate models. f1-score and AUC score are used to assess performance.

# %%
def evaluate_model(y_true, y_pred, y_pred_proba=None):
    f1 = f1_score(y_true, y_pred, average='weighted')
    auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr') if y_pred_proba is not None else None
    return {'F1 Score': f1, 'AUC Score': auc}

# %% [markdown]
# ### Logistic Regression implementation


# %%
if optimise_models:
    lr_param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'max_iter': [1000],
        'solver': ['sag']
    }
    lr_grid = GridSearchCV(
        LogisticRegression(random_state=42, solver='sag', n_jobs=n_jobs),
        lr_param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=n_jobs
    )
    lr_grid.fit(X_train, y_train)
    lr_best = lr_grid.best_estimator_
    print("\nLogistic Regression Best Parameters:")
    print(f"C: {lr_grid.best_params_['C']}")
    print(f"max_iter: {lr_grid.best_params_['max_iter']}")
    print(f"solver: {lr_grid.best_params_['solver']}")
else:
    lr_best = LogisticRegression(C=1.0, max_iter=1000, solver='sag', random_state=42, n_jobs=n_jobs)
    lr_best.fit(X_train, y_train)

lr_pred = lr_best.predict(X_test)
lr_pred_proba = lr_best.predict_proba(X_test)

print("\nLogistic Regression Results:")
if optimise_models:
    print("Best parameters:", lr_grid.best_params_)
print(evaluate_model(y_test, lr_pred, lr_pred_proba))

# %% [markdown]
# The results for LR were quite impressive especially for a simplistic linear model.

# %% [markdown]
# ### KNN implementation

# %%
if optimise_models:
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    knn_grid = GridSearchCV(
        KNeighborsClassifier(n_jobs=n_jobs),
        knn_param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=n_jobs
    )
    knn_grid.fit(X_train, y_train)
    knn_best = knn_grid.best_estimator_
    print("\nKNN Best Parameters:")
    print(f"n_neighbors: {knn_grid.best_params_['n_neighbors']}")
    print(f"weights: {knn_grid.best_params_['weights']}")
    print(f"metric: {knn_grid.best_params_['metric']}")
else:
    knn_best = KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform',
        metric='euclidean',
        n_jobs=n_jobs
    )
    knn_best.fit(X_train, y_train)

knn_pred = knn_best.predict(X_test)
knn_pred_proba = knn_best.predict_proba(X_test)

print("\nKNN Results:")
if optimise_models:
    print("Best parameters:", knn_grid.best_params_)
print(evaluate_model(y_test, knn_pred, knn_pred_proba))

# %% [markdown]
# KNN results were spectacular achieving almost perfect results.

# %% [markdown]
# ### ResNet implementation

# %%
# ResNet implementation with DataParallel
class ResNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, dropout_rate=0.2):
        super(ResNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x):
        identity = self.projection(x)
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out += identity
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

# %% [markdown]
# ### ResNet implementation

# %%
X_train_array = np.array(X_train, dtype=np.float32)
y_train_array = np.array(y_train, dtype=np.int64)
X_test_array = np.array(X_test, dtype=np.float32)
y_test_array = np.array(y_test, dtype=np.int64)

X_train_tensor = torch.from_numpy(X_train_array)
y_train_tensor = torch.from_numpy(y_train_array)
X_test_tensor = torch.from_numpy(X_test_array)
y_test_tensor = torch.from_numpy(y_test_array)

# %%
# training loop
if optimise_models:
    # Define hyperparameter grid
    resnet_params = {
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [32, 64, 128]
    }
    
    best_val_loss = float('inf')
    best_params = None
    best_model = None
    
    # Simple grid search with validation split
    for dropout_rate in resnet_params['dropout_rate']:
        for lr in resnet_params['learning_rate']:
            for batch_size in resnet_params['batch_size']:
                print(f"\nTrying parameters: dropout={dropout_rate}, lr={lr}, batch_size={batch_size}")
                
                # Create validation split
                train_size = int(0.8 * len(X_train_tensor))
                indices = torch.randperm(len(X_train_tensor))
                train_indices = indices[:train_size]
                val_indices = indices[train_size:]
                
                X_train_subset = X_train_tensor[train_indices]
                y_train_subset = y_train_tensor[train_indices]
                X_val = X_train_tensor[val_indices]
                y_val = y_train_tensor[val_indices]
                
                # Create data loaders
                train_dataset = TensorDataset(X_train_subset, y_train_subset)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                val_dataset = TensorDataset(X_val, y_val)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                
                # Initialize model with fixed hidden_dim=128
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = ResNet(X_train_tensor.shape[1], len(np.unique(y_train)), 
                             hidden_dim=128, dropout_rate=dropout_rate).to(device)
                if n_gpus > 1:
                    model = nn.DataParallel(model)
                
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                
                # Training loop
                model.train()
                for epoch in range(5):  # Reduced epochs for faster optimization
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        val_loss += criterion(outputs, batch_y).item()
                
                val_loss /= len(val_loader)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = {
                        'dropout_rate': dropout_rate,
                        'learning_rate': lr,
                        'batch_size': batch_size
                    }
                    best_model = model.state_dict().copy()
    
    print("\nBest ResNet Parameters:")
    print(f"dropout_rate: {best_params['dropout_rate']}")
    print(f"learning_rate: {best_params['learning_rate']}")
    print(f"batch_size: {best_params['batch_size']}")
    
    # Use best parameters for final training
    model = ResNet(X_train_tensor.shape[1], len(np.unique(y_train)), 
                  hidden_dim=128, dropout_rate=best_params['dropout_rate']).to(device)
    if n_gpus > 1:
        model = nn.DataParallel(model)
    model.load_state_dict(best_model)
    
else:
    # Use default parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(X_train_tensor.shape[1], len(np.unique(y_train))).to(device)
    if n_gpus > 1:
        model = nn.DataParallel(model)

# %%
# Train ResNet with final parameters
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001 if not optimise_models else best_params['learning_rate'])

# Create dataset and dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
batch_size = best_params['batch_size'] if optimise_models else 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model.train()
for epoch in range(10):
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# %%
# Evaluate ResNet
model.eval()
with torch.no_grad():
    X_test_tensor = X_test_tensor.to(device)
    logits = model(X_test_tensor)
    resnet_pred_proba = F.softmax(logits, dim=1).cpu().numpy()  # Convert logits to probabilities
    resnet_pred = np.argmax(resnet_pred_proba, axis=1)

print("\nResNet Results:")
print(evaluate_model(y_test_array, resnet_pred, resnet_pred_proba))

# %% [markdown]
# Resnet's results were on par with that of LR.

# %%
all_models = {
    'Logistic Regression': (lr_pred, lr_pred_proba),
    'KNN': (knn_pred, knn_pred_proba),
    'ResNet': (resnet_pred, resnet_pred_proba)
}

# %%
# All models are compared
model_scores = pd.DataFrame([
    {
        'Model': name,
        'F1 Score': evaluate_model(y_test, pred, pred_proba)['F1 Score'],
        'AUC Score': evaluate_model(y_test, pred, pred_proba)['AUC Score']
    }
    for name, (pred, pred_proba) in all_models.items()
])

# %% [markdown]
# KNN was easily the best performing model.

# %%
# Plot F1 scores
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='F1 Score', data=model_scores)
plt.xticks(rotation=45)
plt.title('Model Comparison - F1 Scores')
plt.tight_layout()
plt.show()

# %%
# Plot AUC scores
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='AUC Score', data=model_scores)
plt.xticks(rotation=45)
plt.title('Model Comparison - AUC Scores')
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Task 2: Most Important Features
# 
# Analyse the feature set to identify the most important features for prediction. Methods used are Random Forest and Permutation Importance.

# %% [markdown]
# ### Method 1: Random Forest Feature Importance

# %%
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nRandom Forest Feature Importance:")
print(rf_importance.head(10))

plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=rf_importance.head(10))
plt.title('Top 10 Features - Random Forest Importance')
plt.show()

# %% [markdown]
# The top 6 rank features are clearly the most important features.

# %% [markdown]
# ### Method 2: Permutation Importance

# %%
perm_importance = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42
)
perm_importance_df = pd.DataFrame({
    'feature': selected_features,
    'importance': perm_importance.importances_mean
}).sort_values('importance', ascending=False)

print("\nPermutation Feature Importance:")
print(perm_importance_df.head(10))

plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=perm_importance_df.head(10))
plt.title('Top 10 Features - Permutation Importance')
plt.show()

# %% [markdown]
# The top 5 rank features are clearly the most important features.

# %% [markdown]
# ### Method 3: Consensus Feature Importance
# Averages the rank of the top 10 features from the Random Forest and Permutation Importance methods.

# %%
# Compare feature importance across methods
importance_comparison = pd.DataFrame({
    'feature': selected_features,
    'rf_importance': rf_importance['importance'],
    'perm_importance': perm_importance_df['importance']
})

# Calculate average importance rank across methods
importance_comparison['avg_rank'] = importance_comparison[['rf_importance', 'perm_importance']].rank(ascending=False).mean(axis=1)
importance_comparison = importance_comparison.sort_values('avg_rank')

print("\nConsensus Feature Importance (Average Rank):")
print(importance_comparison.head(10))

plt.figure(figsize=(12, 6))
sns.barplot(x='avg_rank', y='feature', data=importance_comparison.head(10))
plt.title('Top 10 Features - Consensus Importance')
plt.show()

# %% [markdown]
# fwd_URG_flag_count and fwd_last_window_size were clearly the most important features.

# %% [markdown]
# ## Task 3: Ensemble Methods
# In this section we create two ensembles to see how it affects the performance of the models.

# %% [markdown]
# ### Ensemble 1: Using existing models
# Existing models are used and stacked using Logistic Regression as the final estimator.

# %%
# Ensemble 1: Using existing models
print("\nEnsemble 1 Base Models Performance:")
print("Logistic Regression F1 Score:", evaluate_model(y_test, lr_pred, lr_pred_proba)['F1 Score'])
print("KNN F1 Score:", evaluate_model(y_test, knn_pred, knn_pred_proba)['F1 Score'])

# Create stacking ensemble with LogisticRegression as final estimator
ensemble1 = StackingClassifier(
    estimators=[
        ('lr', lr_best),
        ('knn', knn_best)
    ],
    final_estimator=LogisticRegression(random_state=42, n_jobs=n_jobs),
    cv=5,
    n_jobs=n_jobs
)

# Train the first ensemble
start_time = time.time()
ensemble1.fit(X_train, y_train)
ensemble1_time = time.time() - start_time

# Make predictions
ensemble1_pred = ensemble1.predict(X_test)
ensemble1_pred_proba = ensemble1.predict_proba(X_test)

print("\nEnsemble 1 Results (Stacking with Existing Models):")
print("Training time:", ensemble1_time, "seconds")
print(evaluate_model(y_test, ensemble1_pred, ensemble1_pred_proba))

# Print feature importances from the final estimator
if hasattr(ensemble1.final_estimator_, 'coef_'):
    print("\nStacking Final Estimator Feature Importances:")
    for i, coef in enumerate(ensemble1.final_estimator_.coef_[0]):
        print(f"Feature {i+1} importance: {coef:.4f}")

# %% [markdown]
# There was no improvement in the final f1-score of the ensemble over the best performing base model, KNN

# %% [markdown]
# ### Ensemble 2: Using new models
# Three new models are trained using Random Forest, Logistic Regression and SVM. These are then combined using a soft voting classifier.

# %%
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=n_jobs
)

svm = SVC(
    probability=True,
    random_state=42
)

print("\nTraining Ensemble 2 Base Models...")
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_pred_proba = rf.predict_proba(X_test)
svm_pred = svm.predict(X_test)
svm_pred_proba = svm.predict_proba(X_test)

print("\nEnsemble 2 Base Models Performance:")
print("Random Forest F1 Score:", evaluate_model(y_test, rf_pred, rf_pred_proba)['F1 Score'])
print("Logistic Regression F1 Score:", evaluate_model(y_test, lr_pred, lr_pred_proba)['F1 Score'])
print("SVM F1 Score:", evaluate_model(y_test, svm_pred, svm_pred_proba)['F1 Score'])

# Create the second ensemble
ensemble2 = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('lr', LogisticRegression(random_state=42, n_jobs=n_jobs)),
        ('svm', svm)
    ],
    voting='soft',
    n_jobs=n_jobs
)


start_time = time.time()
ensemble2.fit(X_train, y_train)
ensemble2_time = time.time() - start_time

ensemble2_pred = ensemble2.predict(X_test)
ensemble2_pred_proba = ensemble2.predict_proba(X_test)

print("\nEnsemble 2 Results (New Models):")
print("Training time:", ensemble2_time, "seconds")
print(evaluate_model(y_test, ensemble2_pred, ensemble2_pred_proba))

# %% [markdown]
# This ensemble method didnt produce a more predictive model over one of its base models, that being random forest

# %%
# %% [markdown]
# ### Ensemble Comparison
# Both ensembles are compared to see how they perform.

# %%
# Compare both ensembles
ensemble_comparison = pd.DataFrame([
    {
        'Ensemble': 'Existing Models',
        'F1 Score': evaluate_model(y_test, ensemble1_pred, ensemble1_pred_proba)['F1 Score'],
        'AUC Score': evaluate_model(y_test, ensemble1_pred, ensemble1_pred_proba)['AUC Score'],
        'Training Time': ensemble1_time
    },
    {
        'Ensemble': 'New Models',
        'F1 Score': evaluate_model(y_test, ensemble2_pred, ensemble2_pred_proba)['F1 Score'],
        'AUC Score': evaluate_model(y_test, ensemble2_pred, ensemble2_pred_proba)['AUC Score'],
        'Training Time': ensemble2_time
    }
])

print("\nEnsemble Comparison:")
print(ensemble_comparison)

plt.figure(figsize=(12, 6))
sns.barplot(x='Ensemble', y='F1 Score', data=ensemble_comparison)
plt.title('Ensemble Comparison - F1 Scores')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='Ensemble', y='AUC Score', data=ensemble_comparison)
plt.title('Ensemble Comparison - AUC Scores')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='Ensemble', y='Training Time', data=ensemble_comparison)
plt.title('Ensemble Comparison - Training Time (seconds)')
plt.show()

# %%
# Compare ensembles with individual models
all_models = {
    'Logistic Regression': (lr_pred, lr_pred_proba),
    'KNN': (knn_pred, knn_pred_proba),
    'ResNet': (resnet_pred, resnet_pred_proba),
    'Ensemble 1 (Existing)': (ensemble1_pred, ensemble1_pred_proba),
    'Ensemble 2 (New)': (ensemble2_pred, ensemble2_pred_proba)
}

# %%
# Update model comparison
model_scores = pd.DataFrame([
    {
        'Model': name,
        'F1 Score': evaluate_model(y_test, pred, pred_proba)['F1 Score'],
        'AUC Score': evaluate_model(y_test, pred, pred_proba)['AUC Score']
    }
    for name, (pred, pred_proba) in all_models.items()
])

# %%
# Plot graphs
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='F1 Score', data=model_scores)
plt.xticks(rotation=45)
plt.title('Model Comparison - F1 Scores')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='AUC Score', data=model_scores)
plt.xticks(rotation=45)
plt.title('Model Comparison - AUC Scores')
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Task 4: SVM Analysis
# 
# Accessing SVMs for multiclass classification.

# %%
start_time = time.time()

svm = SVC(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    probability=True,
    random_state=42
)

print("\nTraining SVM model...")
svm.fit(X_train, y_train)
svm_training_time = time.time() - start_time

# Make predictions
svm_pred = svm.predict(X_test)
svm_pred_proba = svm.predict_proba(X_test)

print("\nSVM Results:")
print("Training time:", svm_training_time, "seconds")
print(evaluate_model(y_test, svm_pred, svm_pred_proba))

# %% [markdown]
# SVM performance was on par with the least best classifiers however took a lot of time to train.


# %%
# Compare SVM with other models
all_models_with_svm = {
    'Logistic Regression': (lr_pred, lr_pred_proba),
    'KNN': (knn_pred, knn_pred_proba),
    'ResNet': (resnet_pred, resnet_pred_proba),
    'Ensemble 1 (Existing)': (ensemble1_pred, ensemble1_pred_proba),
    'Ensemble 2 (New)': (ensemble2_pred, ensemble2_pred_proba),
    'SVM': (svm_pred, svm_pred_proba)
}

# %%
# Update model comparison
model_scores_with_svm = pd.DataFrame([
    {
        'Model': name,
        'F1 Score': evaluate_model(y_test, pred, pred_proba)['F1 Score'],
        'AUC Score': evaluate_model(y_test, pred, pred_proba)['AUC Score']
    }
    for name, (pred, pred_proba) in all_models_with_svm.items()
])

# %%
# Plot updated F1 scores
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='F1 Score', data=model_scores_with_svm)
plt.xticks(rotation=45)
plt.title('Model Comparison - F1 Scores (Including SVM)')
plt.tight_layout()
plt.show()

# %%
# Plot updated AUC scores
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='AUC Score', data=model_scores_with_svm)
plt.xticks(rotation=45)
plt.title('Model Comparison - AUC Scores (Including SVM)')
plt.tight_layout()
plt.show()

# %% [markdown]
# Compare all models in a table

# %%
performance_data = []
for name, (pred, pred_proba) in all_models_with_svm.items():
    metrics = evaluate_model(y_test, pred, pred_proba)
    performance_data.append({
        'Model': name,
        'F1 Score': f"{metrics['F1 Score']:.4f}",
        'AUC Score': f"{metrics['AUC Score']:.4f}" if metrics['AUC Score'] is not None else 'N/A'
    })

performance_df = pd.DataFrame(performance_data)

performance_df = performance_df.sort_values('F1 Score', ascending=False)

print("\nPerformance Metrics for All Models:")
print("=" * 80)
print(performance_df.to_string(index=False))
print("=" * 80)

print("\nBest Performing Models:")
print(f"Best F1 Score: {performance_df.iloc[0]['Model']} ({performance_df.iloc[0]['F1 Score']})")
print(f"Best AUC Score: {performance_df.loc[performance_df['AUC Score'] != 'N/A', 'Model'].iloc[0]} ({performance_df.loc[performance_df['AUC Score'] != 'N/A', 'AUC Score'].iloc[0]})") 


# %% [markdown]
# The best performing model was KNN.