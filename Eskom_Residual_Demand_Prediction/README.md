
# **Eskom Residual Energy Demand Forecasting Project**

## **Project Overview**

This project analyzes South Africa's residual electricity demand using data provided by Eskom, which includes detailed metrics on energy generation sources and demand over time. The goal is to develop machine learning models that can accurately forecast residual demand, thereby helping Eskom optimize resource allocation, imports, and load shedding measures. This is especially valuable given the increasing variability in renewable energy sources.

**Residual Demand** is the hourly average demand that needs to supplied by all ressources that can be dispatched by Eskom. It depends on several sources of dispatchable generation, including Eskom’s own resources, international imports, independent power producers (IPPs), and interruptible load shedding (IOS).
Normally expressed in Mega Watt(MW)

### **Why This Prediction is Valuable ??**

Accurately forecasting residual demand allows Eskom to:
- **Efficiently manage dispatchable resources** such as thermal and nuclear generation.
- **Schedule independent power producers (IPPs)** and imports strategically.
- **Determine the necessity of interruptible load shedding (IOS)** or load reduction to maintain the balance between supply and demand.

Given the fluctuating nature of renewable energy sources, such as wind and solar, accurate residual demand predictions play a crucial role in ensuring stable electricity supply across South Africa.

## **Table of Contents**
1. Project Overview
2. Data Description
3. Requirements
4. Notebook Summary
5. Feature Engineering
6. Results
7. Conclusion
8. Acknowledgments

## **Data Description**

The dataset was provided by Eskom upon request and contains hourly records of various electricity generation and demand metrics spanning from April 2020 to March 2024. This data allows for in-depth analysis of the dynamics of South Africa’s power generation, including renewable energy contributions, imports and exports, and residual demand.

### **Key Columns in the Dataset**

- **Date_Time**: Timestamp of the data record.
- **Time_Hr**: Hour of the data record.
- **Residual Forecast**: Forecasted residual electricity demand.
- **RSA Contracted Forecast**: Forecasted contracted electricity demand within South Africa.
- **Dispatchable Generation**: Total capacity of dispatchable (controllable) generation sources.
- **Residual Demand**: Actual recorded residual electricity demand.
- **Thermal Generation, Nuclear Generation, Wind, PV, CSP**: Metrics for different energy generation sources.
- **International Exports/Imports**: Values representing international electricity trade (exports and imports).


## **Notebook Summary**

The project notebook includes the following sections:

1. **Data Loading and Preprocessing**:
   - Loaded Eskom’s dataset and performed initial cleaning to handle missing values and standardize features for consistency across variables.

2. **Feature Engineering**:
Combining Date_time and Time_hr column together

Creation of new columns Days_of_the_week, month, hours from the DateTime column

---Cyclic Transformation for Time Features: Since hours and days of the week are cyclical, consider transforming them using sine and cosine transformations. This will capture the cyclic nature of time in a way that linear or tree-based models can leverage:

---Time Features: Extract hour, day of the week, and month as separate columns.

---Converting these to cyclic features (e.g., sin and cos transformations for hour) can help capture time-related patterns.

3. **Model Training and Evaluation**:
   - Trained and evaluated three machine learning models: Random Forest, XGBoost, and CatBoost Regressor, using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score.

   - Plotted **MAE, MSE and R² COMPARISONS PLOTS** comparing the metrics for each model

   - Plotted **Actual vs. Predicted** graphs to visually assess the accuracy of each model's predictions.

     **Feature Importance Analysis**:
   - Analyzed the importance of different features in each model to    understand the factors driving residual demand predictions.

## **Results**

### Model Performance Summary

Three models—**RandomForestRegressor**, **XGBRegressor**, and **CatBoostRegressor**—were trained and evaluated on the dataset. Below is a summary of their performance based on Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score, along with their predicted vs. actual values for the first five records.

1. **RandomForestRegressor**:
   - **MAE**: 183.72
   - **MSE**: 82,206.21
   - **R²**: 0.9886
   - **Predicted vs. Actual (first 5 rows)**:
     ```
               Actual      Predicted
     28051  29035.998  29037.601389
     28052  27200.116  27360.841894
     28053  24121.769  24887.060804
     28054  21982.605  22332.855589
     28055  19751.283  19775.903597
     ```
   - **Observations**: RandomForestRegressor had reasonable performance with an R² of 0.9886, indicating good predictive ability. However, it had the highest MAE and MSE among the three models, suggesting it struggled to capture some of the finer nuances in the data.

2. **XGBRegressor**:
   - **MAE**: 90.66
   - **MSE**: 22,620.86
   - **R²**: 0.9969
   - **Predicted vs. Actual (first 5 rows)**:
     ```
               Actual      Predicted
     28051  29035.998  29107.222656
     28052  27200.116  27315.818359
     28053  24121.769  24249.087891
     28054  21982.605  22041.599609
     28055  19751.283  19758.601562
     ```
   - **Observations**: XGBRegressor showed a substantial improvement over RandomForestRegressor with a lower MAE and MSE and a higher R² of 0.9969. The predictions were closely aligned with actual values, indicating strong performance and a good ability to capture patterns in the data.

3. **CatBoostRegressor**:
   - **MAE**: 66.68
   - **MSE**: 8,839.55
   - **R²**: 0.9988
   - **Predicted vs. Actual (first 5 rows)**:
     ```
               Actual      Predicted
     28051  29035.998  29088.599174
     28052  27200.116  27312.165912
     28053  24121.769  24072.989672
     28054  21982.605  22081.581644
     28055  19751.283  19888.174821
     ```
   - **Observations**: CatBoostRegressor achieved the best performance among the three models, with the lowest MAE and MSE and the highest R² score of 0.9988. The predicted values were nearly identical to the actual values, indicating that CatBoostRegressor captured the underlying data patterns exceptionally well.

### Overall Observations
- **CatBoostRegressor** delivered the most accurate predictions, with the lowest error metrics and a near-perfect R² score, confirming its superior ability to model the data effectively.
- **XGBRegressor** also performed well, with slightly higher error values but still providing highly accurate predictions.
- **RandomForestRegressor** lagged behind in terms of accuracy, with the highest MAE and MSE, indicating it was less effective in capturing detailed patterns in the data compared to the other models.



## GRAPH PLOTS
### **Actual vs. Predicted Analysis**

### Actual vs. Predicted Analysis

1. **CatBoost**:
   - CatBoost predictions closely align with the actual values, showing minimal spread around the 45-degree line.
   - This tight clustering indicates CatBoost’s high accuracy, with the smallest deviations from actual values among the models.

2. **XGBoost**:
   - XGBoost predictions are well-aligned with the actual values and show a tighter clustering around the 45-degree line compared to Random Forest.
   - This suggests XGBoost captures the data pattern better, with fewer large deviations.

3. **RandomForestRegressor**:
   - Random Forest predictions are generally close to the actual values, with most points near the 45-degree line.
   - However, there is some visible spread, indicating a few deviations where the predictions do not perfectly match.



### Feature Importance Analysis

1. **CatBoost**:
   - CatBoost has a balanced feature importance, with multiple features showing high importance.
   - This model considers a wide range of features, making it more capable of capturing complex patterns in the data.

2. **Random Forest**:
   - Random Forest's feature importance is spread across more predictors, although a few features stand out.
   - It uses a broader set of features for prediction, contributing to its overall accuracy.

3. **XGBoost**:
   - XGBoost's feature importance is dominated by a single feature, which it relies on heavily for predictions.
   - This focus on one main predictor makes XGBoost less reliant on a diverse set of features.


## **Conclusion**

This project demonstrated that CatBoost is the most accurate and robust model for predicting residual demand in Eskom's dataset. By examining feature importance and model performance, we identified key factors influencing residual demand, providing valuable insights into how Eskom’s energy demand is influenced by dispatchable generation, imports, exports, and renewable energy contributions.

### **Key Takeaways**:
- **CatBoost** outperformed other models due to its ability to leverage multiple features effectively.

- **Accurate residual demand predictions** allow Eskom to optimize energy dispatch and improve overall system efficiency.
- Understanding feature importance helps Eskom focus on the most impactful variables for resource planning and load balancing.



## **Requirements**

To replicate this project, install the following libraries:

```python
# Core Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

Install using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost
```

## **Acknowledgments**

A special thanks to **Eskom** for providing the data and making this project possible. Accurate energy demand forecasting plays a crucial role in supporting sustainable energy management, and this project provides valuable insights to support Eskom’s operational planning.
