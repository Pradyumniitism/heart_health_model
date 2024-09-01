
# Heart Disease Prediction Project

## Overview

This project aims to predict heart disease using various machine learning algorithms. The dataset used includes key features such as age, gender, cholesterol levels, and chest pain type, among others. The project involves data preprocessing, feature engineering, model selection, and evaluation using several classifiers to determine the best-performing model.

## Project Structure

- **Importing Important Libraries**: Includes necessary libraries such as `pandas`, `numpy`, `scikit-learn`, `seaborn`, and `matplotlib`.
- **Loading the Data**: The dataset is loaded and examined for null values, data types, and overall summary statistics.
- **Data Visualization**: A pair plot and correlation heatmap are created to explore relationships between features and the target variable.
- **Feature Engineering**: Strong features are selected based on correlation with the target variable, and multicollinearity is addressed by dropping highly correlated features.
- **Data Preprocessing**: Categorical features are converted to dummy variables, and all features are scaled using `StandardScaler`.
- **Model Training**: Various classifiers are trained using `GridSearchCV` to find the best hyperparameters. The classifiers and their hyperparameters are as follows:

    - **Logistic Regression**:
      - `penalty`: The regularization method used ('l1' or 'l2').
      - `C`: The regularization strength (values: 0.01, 0.1, 1, 10, 100).
      - `solver`: The optimization algorithm ('liblinear' or 'saga').
      - `max_iter`: The maximum number of iterations for the solver (values: 100, 200, 300).

    - **Support Vector Machine (SVM)**:
      - `C`: The regularization parameter (values: 1, 10, 100).
      - `kernel`: The kernel type ('linear' or 'rbf').
      - `gamma`: The kernel coefficient (values: 'scale' or 'auto').
      - `degree`: The degree of the polynomial kernel function (values: 2, 3, 4).

    - **Gradient Boosting**:
      - `n_estimators`: The number of boosting stages to be run (values: 50, 100, 200).
      - `learning_rate`: The learning rate (values: 0.01, 0.1, 0.2).
      - `max_depth`: The maximum depth of the individual trees (values: 3, 4, 5).

    - **Naive Bayes**:
      - `var_smoothing`: The smoothing parameter to prevent zero probabilities (values: 1e-9, 1e-8, 1e-7).

- **Model Evaluation**: The best models are evaluated on the test set, and their performance is compared using accuracy scores and classification reports.
- **Visualization**: A bar plot with hover functionality is created using Plotly to compare the test accuracies across different models.

## Installation

To run this project, you need to have the following libraries installed:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib plotly
```

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd heart-disease-prediction
   ```
3. Run the script to see the results:
   ```bash
   python heart_disease_prediction.py
   ```

## Results

The project evaluates multiple machine learning models and identifies the best-performing model based on cross-validation accuracy. The results show that [Insert Best Model] achieved the highest accuracy on the test set.

## Visualization

A Plotly bar plot is included to visualize the test accuracies across different classifiers. Hover over each bar to see the exact accuracy.

## Contributing

Feel free to submit issues or pull requests if you want to contribute to this project.
