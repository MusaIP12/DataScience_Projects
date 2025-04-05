# Customer Churn Prediction Dashboard

This project analyzes customer churn for a bank using machine learning. It includes a full pipeline:

-  EDA & Feature Engineering in Jupyter
- 📈 Trained models: Logistic Regression & XGBoost
- 📊 Streamlit Dashboard
-  Live prediction form
-  Interactive filters (Gender & Geography)

## Live App

👉 [Click here to launch the dashboard](https://yourusername.streamlit.app) *(link after Step 7)*

## 📁 Folder Contents

| File                          | Description |
|-------------------------------|-------------|
| `churn_dashboard.py`          | Streamlit dashboard app |
| `churn_dashboard_data.csv`    | Final dataset with features |
| `xgb_model.pkl`               | Trained XGBoost model |
| `Customer_Churn_Notebook.ipynb` | Full Jupyter notebook |
| `original_dataset.csv`        | Original raw dataset |
| `requirements.txt`            | List of Python packages |

##  Run Locally

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run churn_dashboard.py
