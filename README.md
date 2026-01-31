# Credit Card Fraud Detection System

## Project Overview
This project develops a machine learning solution for detecting fraudulent credit card transactions. The system processes and analyzes a dataset of over 1 million transactions to identify patterns associated with financial fraud.

## Technical Specifications
- **Programming Language:** Python
- **Core Libraries:** Pandas, Scikit-learn, XGBoost, Joblib
- **Models Implemented:** Random Forest Classifier, Logistic Regression, XGBoost

## Methodology
1. **Data Preprocessing:** Handled missing values, performed feature engineering by calculating customer age, and applied label encoding to categorical variables such as category, gender, and location.
2. **Model Training:** Utilized an 80/20 split for training and testing. Evaluated multiple algorithms to ensure the highest detection rate.
3. **Model Persistence:** The final Random Forest model and its corresponding feature columns were exported as serialized .pkl files to enable efficient deployment.

## Evaluation Results
The Random Forest model demonstrated the highest performance among the tested algorithms, providing a reliable balance between precision and recall for fraud identification.

## Dataset Source
The dataset used in this project can be found on Kaggle: 
[Credit Card Fraud Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
 
## Implementation
The trained model can be utilized for real-time predictions by loading the saved artifacts. Example implementation:

```python
import joblib
# Load the serialized model
model = joblib.load('fraud_model_rf.pkl')
# Load the feature columns
columns = joblib.load('model_columns.pkl')
