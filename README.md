# Icecream Sales Prediction

## Overview

This project demonstrates the implementation of a simple linear regression model to predict ice cream sales based on temperature. The project is divided into several steps, including data loading, visualization, model training, prediction, and evaluation.

## Table of Contents

1. [Understanding the Problem](#understanding-the-problem)
2. [Importing Libraries and Loading Data](#importing-libraries-and-loading-data)
3. [Visualizing the Data](#visualizing-the-data)
4. [Splitting the Data into Training and Testing Sets](#splitting-the-data-into-training-and-testing-sets)
5. [Training the Linear Regression Model](#training-the-linear-regression-model)
6. [Making Predictions](#making-predictions)
7. [Evaluating the Model](#evaluating-the-model)

## Understanding the Problem

The goal of this project is to predict the number of ice creams sold based on the temperature. We will use a linear regression model to establish a relationship between temperature and ice cream sales.

## Importing Libraries and Loading Data

We start by importing necessary libraries such as `numpy`, `pandas`, `matplotlib`, and `scikit-learn`. We then load a sample dataset containing temperature and ice cream sales data.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset
data = {'Temperature': [18, 22, 27, 30, 35],
        'Ice Creams Sold': [80, 150, 200, 300, 350]}
df = pd.DataFrame(data)
print(df)
```

## Visualizing the Data

We visualize the relationship between temperature and ice cream sales using a scatter plot.

```python
plt.scatter(df['Temperature'], df['Ice Creams Sold'], color='blue')
plt.title('Temperature vs. Ice Creams Sold')
plt.xlabel('Temperature (°C)')
plt.ylabel('Ice Creams Sold')
plt.show()
```

## Splitting the Data into Training and Testing Sets

We split the data into features (X) and target variable (y), and then further split it into training and testing sets.

```python
X = df[['Temperature']]
y = df['Ice Creams Sold']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Training the Linear Regression Model

We create a linear regression model and train it using the training data.

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

## Making Predictions

We use the trained model to predict ice cream sales for the test set and compare the predicted values with the actual values.

```python
y_pred = model.predict(X_test)
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)
```

## Evaluating the Model

We evaluate the model's performance by calculating the Mean Squared Error (MSE).

```python
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
```

## Conclusion

This project provides a basic understanding of how to implement a linear regression model to predict ice cream sales based on temperature. The model's performance is evaluated using Mean Squared Error, and the results are visualized to understand the relationship between the variables.

## Requirements

- Python 3.x
- Libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`

## Installation

To install the required libraries, run the following command:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

1. Clone the repository.
2. Open the Jupyter notebook `Project04_Linear_Regression.ipynb`.
3. Run each cell to execute the code and see the results.

