# Walmart Weekly Sales Prediction
This project analyzes and predicts Walmart's weekly sales using historical data. We explore seasonal patterns, holiday impacts, and other key variables to understand trends and improve forecasting accuracy. The model used is a RandomForestRegressor, known for handling non-linear relationships in the data, providing a robust prediction for weekly sales across Walmart stores.

## Table of Contents

1.Project Overview 

2.Data 

3.Features

4.Exploratory Data Analysis

5.Modeling and Evaluation

6.Results

7.Installation and Setup

8.Usage

9.Tools

10.Contributors

## 1.Project Overview

The objective is to forecast weekly sales for Walmart stores and identify how factors like holidays, temperature, fuel prices, and economic indicators (CPI and Unemployment) influence sales trends. By improving predictions, we help Walmart better manage inventory, pricing, and staffing.

## 2.Data

The dataset consists of weekly sales data for Walmart stores, with variables including:

Weekly_Sales: Sales amount for the store that week

Holiday_Flag: Whether that week contained a holiday

Temperature: Temperature in the area

Fuel_Price: Cost of fuel in the area

CPI: Consumer Price Index

Unemployment: Area unemployment rate


## 3.Features

Date preprocessing and feature engineering: Year, Month, and Day columns for time-based analysis.

Aggregated Sales: Weekly sales summed across stores and dates to get a clear overview.

Handling Missing Values: Checked for any NaN values and filled/removed as necessary.

Feature Selection: Selected relevant features (Store, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment, Year, Month, Day) for training the model.
## 4.Exploratory Data Analysis

In-depth analysis of data to identify trends:

Sales vs. Month: Sales patterns across months, highlighting seasonal spikes in Nov-Dec.

Daily Sales Trends: Plots of daily sales show fluctuations over time.

Holiday vs. Non-Holiday Sales: Comparison to see how holidays affect sales.
## 5.Modeling and Evaluation

The RandomForestRegressor model was chosen for its flexibility in handling complex relationships:

Data Splitting: Chronologically split data into training (before 2012) and test sets (2012 and later).

Model Training: The model was trained on historical weekly sales data.

Model Evaluation: Evaluated using Mean Absolute Error (MAE) to understand prediction accuracy.
## 6.Results

MAE: The model achieved a Mean Absolute Error of approximately 176,392, indicating predictions are around 12.4% off the typical weekly sales amount.

Visualizations:

Actual vs. Predicted Sales: Line and bar plots to compare real vs. predicted sales.

Monthly Sales Patterns: Bar charts showing high sales in November-December.
## 7.Installation and Setup

To run the code on your local machine, clone the repository and install dependencies:

git clone https://github.com/yourusername/Walmart-Sales-Prediction.git
cd Walmart-Sales-Prediction
pip install -r requirements.txt
## 8.Usage

Data Loading and Preprocessing: Load and preprocess the data with data_load.py.

Exploratory Analysis: Run exploratory_analysis.ipynb for initial insights and visualization.

Model Training and Evaluation: Train the model and evaluate performance by running model_training.py.

Visualization: Generate plots for actual vs. predicted sales by running visualizations.py.
## 9.Tools

This project uses various tools and libraries for data analysis, modeling, and visualization:

#### Data Manipulation:

Pandas: For data cleaning, manipulation, and preprocessing.

NumPy: For numerical operations, including handling missing values and outlier replacement.
#### Visualization:

Matplotlib: Used for basic visualizations and plotting the trends in sales data.

Seaborn: Provides enhanced visualizations for comparing sales on holidays vs non-holidays, as well as monthly and seasonal sales distributions.
#### Data Analysis and Statistical Testing:

SciPy: Specifically scipy.stats for calculating correlation coefficients between features and the target variable.

Statsmodels: Used for seasonal decomposition, which helps in analyzing and separating trend, seasonality, and residual components in weekly sales data.

#### Machine Learning:

Scikit-Learn:
RandomForestRegressor: A flexible, non-linear regression model used for predicting weekly sales.

train_test_split: For dividing data into training and test sets, ensuring model validity.

mean_absolute_error: A metric used to evaluate model accuracy by calculating the average absolute difference between predicted and actual values.
#### Warnings Handling:

Warnings Library: To ignore specific warnings that might clutter output, such as deprecations or minor API changes.

## 10.Contributors

Saikiran Barma (saikiranbarma@gmail.com)
