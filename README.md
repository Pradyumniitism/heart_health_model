# Heart Disease Prediction Project

## Overview

This project aims to predict heart disease using various machine learning algorithms. The dataset used includes key features such as age, gender, cholesterol levels, and chest pain type, among others. The project involves data preprocessing, feature engineering, model selection, and evaluation using several classifiers to determine the best-performing model.

## Project Structure

- **Importing Important Libraries**: Includes necessary libraries such as `pandas`, `numpy`, `scikit-learn`, `seaborn`, and `matplotlib`.
- **Loading the Data**: The dataset is loaded and examined for null values, data types, and overall summary statistics.
- **Data Visualization**: A pair plot and correlation heatmap are created to explore relationships between features and the target variable.
- **Feature Engineering**: Strong features are selected based on correlation with the target variable, and multicollinearity is addressed by dropping highly correlated features.
- **Data Preprocessing**: Categorical features are converted to dummy variables, and all features are scaled using `StandardScaler`.
- **Model Training**: Various classifiers, including Logistic Regression, SVM, Gradient Boosting, and Naive Bayes, are trained using `GridSearchCV` to find the best hyperparameters.
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
