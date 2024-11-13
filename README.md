
# Data Science Projects

This repository contains a collection of data science and machine learning projects that explore various aspects of predictive modelling, classification, natural language processing (NLP), and medical diagnostics. Each project is written in Python using popular libraries such as Pandas, NumPy, Scikit-learn, XGBoost, and more. Below is a summary of the top projects, including two key healthcare-focused projects.

## Top Projects:


## 1. **Eskom Residual Energy Demand Forecasting**

- **Objective**: Forecast South Africa’s residual electricity demand to optimize Eskom’s resource management and load shedding strategies.

- **Models**: Random Forest, XGBoost, CatBoost Regressors

- **Best Model Accuracy:** CatBoost Regressor with R² score of 0.9988

- **Description:** This project leverages machine learning to forecast residual energy demand based on Eskom’s energy generation and demand data. Using feature engineering and model evaluation techniques, the **CatBoost Regressor** 
    was identified as the most effective model, accurately capturing demand patterns influenced by renewable energy sources, imports, exports, and dispatchable generation. 
    
### 2. **COVID-19 Classification Using Audio Features**
   - **Objective**: Classify COVID-19 cases based on audio recordings (cough, breath, and speech).
   - **Models**:
     - **Random Forest**: 
       - Accuracy: 95.4%
     - **Support Vector Machine (SVM)**: 
       - Accuracy: 93.7%
   - **Description**: This project applies audio signal processing techniques to extract features (MFCC, ZCR, spectral features) from audio recordings to detect COVID-19. It demonstrates the use of machine learning models in healthcare by utilizing non-invasive audio data for classification.


### 3. **Brain Tumor Image Classification**
   - **Objective**: Detect and classify brain tumors from medical images using machine learning.
   - **Model**: Convolutional Neural Network (CNN)
     - Accuracy: 98.75%
   - **Description**: This project focuses on using image processing and deep learning techniques to classify brain tumors based on MRI scans. A CNN model was trained to recognize different tumor types with high accuracy, demonstrating the potential of AI in medical diagnostics.
---

## Other Projects:

### 3. **Iris Flower Classification**
   - **Objective**: Classify different species of Iris flowers based on petal and sepal measurements.
   - **Model**: Decision Tree Classifier
     - Validation Accuracy: 96.67%
     - Test Accuracy: 93.33%
   - **Description**: A simple yet effective project that demonstrates classification tasks using decision trees on the classic Iris flower dataset.

### 4. **Email Spam Detection (NLP Project)**
   - **Objective**: Detect whether an email is spam or not by analyzing its content.
   - **Models**:
     - **Decision Tree Classifier**:
       - Validation Accuracy: 97.13%
       - Test Accuracy: 96.77%
     - **Naive Bayes Classifier**:
       - Validation Accuracy: 96.17%
       - Test Accuracy: 96.89%
   - **Description**: This project uses Natural Language Processing (NLP) to classify emails as spam or not spam, showcasing how text-based data can be effectively classified using machine learning.

### 5. **Sales Prediction**
   - **Objective**: Predict sales numbers based on advertising spending across different media (TV, radio, and newspaper).
   - **Models**:
     - **Decision Tree Regressor**:
       - Validation MAE: 1.0775
       - Test MAE: 0.935
     - **XGBoost**:
       - Validation MAE: 0.7401
       - Test MAE: 0.7137
   - **Description**: This regression project predicts sales using advertising data. The XGBoost model achieved the lowest mean absolute error, demonstrating its efficiency in predictive tasks.

### 6. **Car Price Prediction**
   - **Objective**: Predict car prices based on several factors, including engine size, mileage, and brand.
   - **Model**: Decision Tree Regressor
     - Validation Prediction Error: 1.179
     - Test Prediction Error: 1.5698
   - **Description**: This project uses decision tree regression to predict car prices, illustrating how machine learning models can assist in pricing tasks based on numerical and categorical features.

---

## Libraries and Tools Used:
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computing.
- **Seaborn** and **Matplotlib**: Data visualization.
- **Scikit-learn**: Machine learning algorithms and preprocessing.
- **XGBoost**: Gradient boosting for regression and classification.
- **NLTK**: Natural Language Toolkit for NLP in the spam detection project.
- **Librosa**: Audio processing library used in the COVID-19 classification project.
- **TensorFlow/Keras**: Deep learning frameworks used for brain tumor classification.

## Setup and Installation:
To run any of these projects locally, ensure that Python is installed and run the following command to install the required libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost nltk librosa tensorflow keras
```

---

Each project in this repository is documented in Jupyter Notebooks, allowing for easy understanding and reproduction of the experiments. You can explore the projects to gain insights into various machine-learning techniques and their real-world applications, especially in the healthcare domain.


