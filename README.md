# Mental Health Classification Project

This repository contains a Jupyter notebook that analyzes a mental health dataset to predict mood swings and treatment outcomes. The project demonstrates various machine learning techniques for classification tasks on healthcare data.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Concepts Explained](#key-concepts-explained)
- [Results](#results)
- [Files](#files)

## Project Overview

This project implements multiple classification models to predict mental health outcomes based on various features including demographic information, health history, and lifestyle factors. Two main classifications are performed:
1. Predicting whether a person receives treatment (first model)
2. Predicting mood swings severity (second model)

## Dataset

The dataset contains various mental health indicators including:
- Demographic information (Gender, Country, Occupation)
- Employment status (self_employed)
- Mental health history (family_history, Mental_Health_History)
- Behavioral factors (Days_Indoors, Growing_Stress, Changes_Habits)
- Current mental state indicators (Mood_Swings, Coping_Struggles, Work_Interest, Social_Weakness)
- Healthcare interactions (mental_health_interview, treatment, care_options)

## Methodology

The notebook demonstrates a complete machine learning pipeline including:

1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical variables
   - Feature engineering from text data
   - Standardization of numerical features

2. **Feature Selection and Engineering**
   - Statistical feature selection
   - Text feature extraction from care options
   - One-hot encoding for categorical variables

3. **Model Development**
   - Multiple classification algorithms
   - Hyperparameter tuning
   - Model evaluation

## Key Concepts Explained

### Data Processing Techniques

#### Label Encoding
```python
label_enc = LabelEncoder()
for col in categorical_cols:
    if col in data.columns:
        if data[col].dtype == 'object' or data[col].dtype == 'bool':
            data[col] = label_enc.fit_transform(data[col].astype(str))
```
**Explanation**: Label encoding converts categorical text values into numerical values. For example, "Yes"/"No" might be converted to 1/0. This is necessary because machine learning algorithms require numerical inputs.

#### One-Hot Encoding
```python
data = pd.get_dummies(data, columns=['Days_Indoors'], prefix='days')
```
**Explanation**: One-hot encoding creates binary columns for each category. For example, if "Days_Indoors" has values like "1-14 days", "15-30 days", it creates separate columns like "days_1-14_days" = 1 or 0. This avoids creating false ordinal relationships between categories.

#### Text Processing
```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text
```
**Explanation**: Text cleaning normalizes text data by converting to lowercase and removing special characters to improve feature extraction.

#### TF-IDF Vectorization
```python
tfidf = TfidfVectorizer(max_features=500)
tfidf_features = tfidf.fit_transform(data['Cleaned_Text']).toarray()
```
**Explanation**: Term Frequency-Inverse Document Frequency is a numerical statistic that reflects how important a word is to a document. It converts text data into meaningful numerical features by weighting terms based on their frequency and rarity across documents.

### Machine Learning Concepts

#### Feature Selection
```python
selector = SelectKBest(score_func=f_classif, k=10)
```
**Explanation**: This selects the most relevant features based on their statistical relationship with the target variable, which helps improve model efficiency and reduce overfitting.

#### Standardization
```python
scaler = StandardScaler()
X_selected = scaler.fit_transform(X_selected)
```
**Explanation**: Standardization scales numerical features to have zero mean and unit variance, which ensures that no feature dominates the model training due to differences in scale.

#### Model Evaluation Metrics

- **Accuracy**: The proportion of correct predictions out of all predictions
- **Classification Report**: Includes precision, recall, F1-score, and support for each class
  - **Precision**: The proportion of true positive predictions among all positive predictions
  - **Recall**: The proportion of true positive predictions among all actual positives
  - **F1-score**: The harmonic mean of precision and recall

#### Model Algorithms

1. **Random Forest**
   ```python
   RandomForestClassifier(random_state=42)
   ```
   **Explanation**: An ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes from individual trees. It's robust against overfitting and handles various feature types well.

2. **Gradient Boosting**
   ```python
   GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5)
   ```
   **Explanation**: A boosting algorithm that builds trees sequentially, with each tree correcting the errors of the previous one. It often achieves high accuracy but can be prone to overfitting without proper tuning.

3. **Logistic Regression**
   ```python
   LogisticRegression(max_iter=1000)
   ```
   **Explanation**: A statistical model that uses a logistic function to model the probability of a binary outcome. It's simpler than tree-based models but provides good interpretability.

4. **Decision Tree**
   ```python
   DecisionTreeClassifier()
   ```
   **Explanation**: A model that predicts the value of a target variable by learning decision rules from features. It's easily interpretable but often needs ensemble methods to improve performance.

#### Hyperparameter Tuning
```python
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
```
**Explanation**: Grid search is a method for finding the optimal combination of hyperparameters for a model by testing all possible combinations from specified parameter ranges.

## Results

The notebook evaluates multiple models and selects the best performer based on accuracy. The final models are saved and their predictions exported to CSV files. 

Two sets of predictions are made:
1. `predictions.csv` - Contains treatment predictions from Random Forest and Gradient Boosting models
2. `predictionss.csv` - Contains mood swings predictions

## Files

- `mental-health-classification.ipynb`: The main Jupyter notebook
- `predictions.csv`: Predictions for treatment outcomes
- `predictionss.csv`: Predictions for mood swings

## Usage

This notebook serves as a comprehensive example for mental health data analysis and can be adapted for similar healthcare classification problems. The techniques demonstrated are applicable to various machine learning tasks involving categorical data, text processing, and classification.
