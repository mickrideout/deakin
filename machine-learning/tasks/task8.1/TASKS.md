# Auto-ML Model Comparison Implementation

This task involves creating a comprehensive machine learning model comparison system using Python, focusing on student performance prediction using various classification models.

## Completed Tasks

- [x] Initial task planning and documentation

## In Progress Tasks

- [ ] Create initial auto-ml.py file with Jupyter annotations
- [ ] Implement data loading and preprocessing pipeline

## Future Tasks

### Data Preprocessing
- [ ] Load Dataset3.csv into pandas DataFrame
- [ ] Handle missing values and 'nan' targets
- [ ] Implement StandardScaler for feature scaling
- [ ] Apply SMOTE for class imbalance
- [ ] Perform feature selection (top 20 features)
- [ ] Encode target variable using LabelEncoder
- [ ] Create train/test split (80/20)

### Task Set 1: PCA Analysis
- [ ] Print dataset dimensions
- [ ] Implement PCA with 3 components
- [ ] Visualise PCA results

### Task Set 2: SVM Models
- [ ] Create first SVM model
- [ ] Create second SVM model
- [ ] Create third SVM model
- [ ] Compare model performances
- [ ] Document hyperparameters

### Task Set 4: KNN Models
- [ ] Create first KNN model
- [ ] Create second KNN model
- [ ] Compare with SVM results

### Task Set 5: KNN Distance Metrics
- [ ] Implement KNN with different distance metrics
- [ ] Compare distance metric performances
- [ ] Document findings and analysis

## Implementation Plan

### Data Processing Pipeline
1. Data loading and cleaning
2. Feature scaling and selection
3. Target encoding and balancing
4. Train/test splitting

### Model Development
1. PCA analysis for dimensionality reduction
2. SVM model development with different parameters
3. KNN implementation with various configurations
4. Performance comparison across all models

### Evaluation Metrics
- AUC score
- F1 score
- Model comparison visualisations

### Relevant Files

- `auto-ml.py` - Main implementation file with Jupyter annotations
- `Dataset3.csv` - Input dataset
- `TASKS.md` - Task tracking and documentation

## Technical Components

### Required Libraries
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- imblearn (for SMOTE)

### Environment Configuration
- Python 3.x
- Jupyter notebook support
- Required package installations

## Data Structure
- 4424 rows × 37 columns
- Mixed data types (int64, float64, object)
- Target variable: 'Target' (categorical) 