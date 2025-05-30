# AI Hackathon: Company Success Prediction

This project was developed during an AI hackathon to predict the success probability of companies (ranging from 0 to 1).  
Using various financial and operational indicators, the model quantifies the likelihood of future success.

## Project Overview
This project uses the XGBoost regression model to predict company success probabilities based on diverse features.  
Optuna was used to tune hyperparameters, and the evaluation metric was Mean Absolute Error (MAE).  
The model was trained and validated using **Stratified K-Fold Cross Validation**, which preserves the distribution of the target variable across all folds for more reliable evaluation.

## Competition Results
| Leaderboard          | Score (Weighted MAE) | Rank        |
|:----------:|:-------------:|-------------|
| Public  | 0.20972      | 23 / 593    |
| Private  | 0.21184      | 23 / 592  |

## Data Preprocessing
The original dataset includes the following features:
- Country  
- Sector  
- Year of Establishment  
- Investment Stage  
- Number of Employees  
- Number of Customers (in millions)  
- Total Investment (in 100M KRW)  
- Annual Revenue (in 100M KRW)  
- Number of SNS Followers (in millions)  
- Company Valuation (in 100M KRW)  
- Acquisition Status  
- IPO Status  

Key preprocessing steps are as follows:
- Converted year of establishment into company age (`2025 - establishment year`)  
- Applied LabelEncoder to categorical variables (Country, Sector)  
- Mapped Investment Stage to ordered integers (Seed=0 to IPO=4)  
- Converted boolean fields (Acquisition/IPO) to 0 and 1  
- Parsed company valuation from string formats like `"6000+"`, `"2500-3500"` to numerical values  
- Missing values were not explicitly imputed and may remain in the model inputs

## Feature Engineering
Several new features were engineered to enhance predictive power, including:
- Investment per employee  
- Revenue per employee  
- Revenue-to-investment ratio  
- Revenue-to-customer ratio  
- Valuation-to-investment ratio  
- Revenue per SNS follower  
- Square root of company age  
- Revenue compared to national/sector averages  

A total of 17 derived features were created. Only a subset was included in the final model based on performance impact.

## Features Used in Final Model
The following features were included in the final training set:
- Company age  
- Investment per employee  
- Revenue per employee  
- Revenue-to-investment ratio  
- Investment-to-valuation ratio  
- Revenue-to-customer ratio  
- Root of company age  
- Valuation-to-investment ratio  
- Followers per customer  
- Followers per valuation  
- Customers per employee  
- Revenue-to-valuation ratio  
- Revenue-to-age ratio  
- Revenue per follower  
- Revenue-to-national-average ratio  
- Customers-to-sector-average ratio

## Removed Features
The following features were excluded due to low predictive power or multicollinearity:
- Country  
- Sector  
- Investment Stage  
- Acquisition Status  
- IPO Status  
- Number of Employees  
- Number of SNS Followers  
- Annual Revenue  
- Revenue-to-total-investment ratio  
- Valuation per employee  
- Investment per age  
- Customers per age  
- Revenue per customer per employee

## Model Training and Tuning
Hyperparameter optimization was conducted using Optuna over 300 trials with MAE as the evaluation metric.  
The model was trained and validated using **Stratified K-Fold Cross Validation**, ensuring balanced distribution of the target variable in each fold.  
XGBoost was trained using `xgb.train` with DMatrix input, and early stopping was applied with a patience of 50 rounds.
