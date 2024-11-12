**Eskom Residual Demand Analysis Project**

This project analyses South Africa's residual demand using data provided by Eskom which includes various energy generation sources and demand metrics over time.

**Residual Demand** depends on several sources of dispatchable generation, including Eskom’s own resources, international imports, independent power producers (IPPs), and interruptible load shedding (IOS).

#### So Why This Prediction is Valuable
Accurately forecasting Residual Demand can help Eskom manage its dispatchable resources more efficiently, schedule IPPs and imports, and decide when IOS or load reduction measures are necessary to balance supply and demand. This is especially important given how fluctuating renewable energy sources are.

Table of Contents
Project Overview
Data Description
Requirements
Notebook Summary
Feature Engineering
Results
Conclusion
Acknowledgments


Project Overview
The purpose of this project is to conduct a thorough analysis of South Africa's residual electricity demand using Eskom's dataset, which includes data on different forms of electricity generation and contracted and residual forecasts. This analysis helps in understanding the dynamics of power generation, imports and exports, and the utilization of renewable energy.

Data Description

The data was provided Eskom upon request.The dataset was in a form of an csv file includes hourly records of various electricity generation metrics from April 2020 to March 2024.

Key Columns:
**Date_Time: Date of data record**
**Time_Hr: Hour of data record**
**Residual Forecast: Forecasted residual electricity demand**
**RSA Contracted Forecast:** Forecast for contracted electricity demand in South Africa
**Dispatchable Generation:** Dispatchable generation capacity
**Residual Demand:** Actual residual electricity demand
**Thermal Generation, Nuclear Generation, Wind, PV, CSP:** Different energy generation sources
**International Exports/Imports:** Export and import values for international electricity trade




Models Used
The following regression models were trained and evaluated:

RandomForestRegressor: An ensemble model based on decision trees.
XGBRegressor: A gradient boosting model using XGBoost.
CatBoost Regressor: A gradient boosting model by Yandex, optimized for categorical and numerical data.


Performance Evaluation
Performance metrics used to evaluate model accuracy:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
R² Score




Results Summary
Model	MAE	MSE	
𝑅
2
R 
2
  Score
RandomForestRegressor	183.72	82,206.21	0.9886
XGBRegressor	90.66	22,620.86	0.9969
CatBoost Regressor	66.68	8,839.55	0.9988
Note: The CatBoost Regressor achieved the best performance, with an 
𝑅
2
R 
2
  score of 0.9988, indicating that it explains 99.88% of the variance in the residual demand.



Discussion:

The CatBoost model had the lowest error metrics (MAE and MSE) and the highest R² score, suggesting it’s the most accurate model for this dataset.
XGBoost also performed well, though it had slightly higher errors than CatBoost.
Random Forest showed the highest MAE and MSE values, making it the least accurate of the three models.
Feature Importance Insights:

All models identified Residual Forecast as a crucial feature.
CatBoost leveraged a broader set of features than the other models, which may explain its superior performance.