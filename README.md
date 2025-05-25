# AI Hackathon: Company Success Prediction

This project was developed during an AI hackathon to predict the success probability of companies (ranging from 0 to 1).  
Using various financial and operational indicators, the model quantifies the likelihood of future success.

#### Project Overview:
This project uses the XGBoost regression model to predict company success probabilities based on diverse features. Optuna was used to tune hyperparameters, and the evaluation metric was Mean Absolute Error (MAE). The model was trained using a hold-out validation strategy via `train_test_split`.

#### Data Preprocessing:
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
- Year of Establishment was transformed into company age (current year - establishment year).  
- Categorical variables (Country, Sector) were encoded using LabelEncoder.  
- Investment Stage was ordinally mapped (e.g., Seed=0 to IPO=4).  
- Boolean values (Acquisition/IPO status) were converted to 0/1.  
- Company Valuation strings (e.g., "6000 이상", "2500-3500") were cleaned into numerical values.  
- Numerical missing values were not explicitly imputed.

#### Feature Engineering:
A wide variety of derived features were created to improve model performance. Examples include:
- Investment per Employee = Total Investment ÷ Number of Employees  
- Revenue per Employee = Annual Revenue ÷ Number of Employees  
- Revenue to Investment Ratio = Annual Revenue ÷ Total Investment  
- Revenue to Customer Ratio = Annual Revenue ÷ Number of Customers  
- Company Valuation to Investment Ratio, Revenue to Valuation Ratio, etc.  
- Root of Company Age, Revenue per Social Follower, and more  
- National and industry-level normalization features (e.g., Revenue ÷ Average of Country)

A total of 17 new features were engineered. Only those that contributed meaningfully to performance were included in the final model.

#### Features Used in Final Model:
The final model was trained using the following features:
- Company Age  
- Investment per Employee  
- Revenue per Employee  
- Revenue to Investment Ratio  
- Investment to Valuation Ratio  
- Revenue to Customer Ratio  
- Root of Company Age  
- Company Valuation to Investment Ratio  
- Social Followers to Customers  
- Social Followers to Valuation  
- Customers per Employee  
- Revenue to Valuation Ratio  
- Revenue to Company Age  
- Revenue per Follower  
- Revenue to National Average Ratio  
- Customers to Sector Average Ratio

#### Removed Features:
The following features were removed due to low impact or multicollinearity:
- Country  
- Sector  
- Investment Stage  
- Acquisition Status  
- IPO Status  
- Number of Employees  
- Number of SNS Followers  
- Annual Revenue  
- Revenue to Total Investment Ratio  
- Valuation per Employee  
- Investment per Age  
- Customers per Age  
- Revenue per Customer per Employee

#### Model Training and Tuning:
Optuna was used to tune hyperparameters through 300 trials, minimizing MAE.  
Data was split using `train_test_split` (80:20), and training was conducted with `xgb.train` using DMatrix format.  
Early stopping was set to 50 rounds to prevent overfitting.

#### Final Results:
The final model achieved a MAE of approximately 0.2043.  
Key contributing features included those based on customer count, employee count, company age, and follower-based ratios.  
Interestingly, the original "Annual Revenue" feature negatively impacted performance and was excluded from the final model, possibly due to overfitting concerns.
