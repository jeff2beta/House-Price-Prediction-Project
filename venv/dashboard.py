import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# Setup logging
logging.basicConfig(filename='dashboard.log', level=logging.INFO)

# Load data and model
data = pd.read_csv('house-prices/train.csv')
model = joblib.load('best_random_forest_model.pkl')

# Load column names and remove 'SalePrice'
with open('columns.txt', 'r') as f:
    columns = f.read().splitlines()
columns.remove('SalePrice')

# Add a new feature 'TotalArea' to the dataset
data['TotalArea'] = data['GrLivArea'] + data['TotalBsmtSF']

# Title
st.title('House Price Prediction Dashboard')

# Sidebar for user input
st.sidebar.header('User Input Features')
def user_input_features():
    TotalArea = st.sidebar.slider('Total Area (Above Ground + Basement)', 500, 8000, 2500)
    OverallQual = st.sidebar.slider('Overall Quality', 1, 10, 5)
    YearBuilt = st.sidebar.slider('Year Built', 1900, 2020, 2000)
    GarageCars = st.sidebar.slider('Number of Garage Cars', 0, 4, 2)
    data = {'TotalArea': TotalArea,
            'OverallQual': OverallQual,
            'YearBuilt': YearBuilt,
            'GarageCars': GarageCars}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Preprocess input
input_df['TotalArea'] = np.log1p(input_df['TotalArea'])
input_df = pd.get_dummies(input_df)

# Ensure all dummy variables are present
missing_cols = set(columns) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0

input_df = input_df[columns]

# Prediction
try:
    prediction = model.predict(input_df)
    prediction_exp = np.expm1(prediction)
    st.subheader('Prediction')
    st.write(f'The predicted house price is: ${prediction_exp[0]:,.2f}')
    logging.info(f'Prediction made: {prediction_exp[0]:,.2f}')
except Exception as e:
    st.error(f'Error making prediction: {e}')
    logging.error(f'Error making prediction: {e}')

# Display correlation matrix
st.subheader('Correlation Matrix')
numeric_cols = data.select_dtypes(include=['number'])
correlation_matrix = numeric_cols.corr()
fig, ax = plt.subplots(figsize=(20, 16))  
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", mask=mask, ax=ax)
plt.title('Correlation Matrix')
st.pyplot(fig)

# Display feature importance
st.subheader('Feature Importance')
importances = model.feature_importances_
if len(columns) != len(importances):
    st.error("Mismatch in feature importance data length.")
else:
    feature_importance_df = pd.DataFrame({'Feature': columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    fig2 = px.bar(feature_importance_df.head(10), x='Importance', y='Feature', orientation='h', title='Top 10 Feature Importances')
    st.plotly_chart(fig2)

# Model performance metrics
st.subheader('Model Performance Metrics')

# Data cleaning: Check for NaNs and infinite values
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Separate numeric and non-numeric columns
numeric_data = data.select_dtypes(include=[np.number])
non_numeric_data = data.select_dtypes(exclude=[np.number])

# Handle only numeric data for finite check
numeric_data = numeric_data.applymap(lambda x: np.nan if not np.isfinite(x) else x)

# Fill NaN values for numeric columns with column mean
numeric_data.fillna(numeric_data.mean(), inplace=True)

# Fill NaN values for non-numeric columns with a placeholder or mode
non_numeric_data.fillna('Missing', inplace=True)

# Concatenate the numeric and non-numeric data back together
data = pd.concat([numeric_data, non_numeric_data], axis=1)

# Ensure there are no missing values in critical columns
critical_columns = ['SalePrice', 'TotalArea', 'OverallQual', 'YearBuilt', 'GarageCars']
data.dropna(subset=critical_columns, inplace=True)

X = data.drop(columns=['SalePrice'])
y = np.log1p(data['SalePrice'])  # Log-transform SalePrice for training
X = pd.get_dummies(X).reindex(columns=columns, fill_value=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model again with all the data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Logging y_pred and y_test for debugging
logging.info(f'y_pred head: {y_pred[:5]}')
logging.info(f'y_test head: {y_test[:5]}')

# Transforming predictions and actual values back from logarithmic scale
y_pred_exp = np.expm1(y_pred)
y_test_exp = np.expm1(y_test)

# Logging transformed values for debugging
logging.info(f'y_pred_exp head after expm1: {y_pred_exp[:5]}')
logging.info(f'y_test_exp head after expm1: {y_test_exp[:5]}')

# Ensure all values are finite for evaluation
y_pred_exp = np.nan_to_num(y_pred_exp, nan=0.0, posinf=0.0, neginf=0.0)
y_test_exp = np.nan_to_num(y_test_exp, nan=0.0, posinf=0.0, neginf=0.0)

mae = mean_absolute_error(y_test_exp, y_pred_exp)
mse = mean_squared_error(y_test_exp, y_pred_exp)
r2 = r2_score(y_test_exp, y_pred_exp)

st.write(f'Mean Absolute Error: {mae:.2f}')
st.write(f'Mean Squared Error: {mse:.2f}')
st.write(f'R-squared: {r2:.2f}')

logging.info(f'Mean Absolute Error: {mae}')
logging.info(f'Mean Squared Error: {mse}')
logging.info(f'R-squared: {r2}')

# Residual plot
st.subheader('Residual Plot')
residuals = y_test_exp - y_pred_exp
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=y_pred_exp, y=residuals, mode='markers', name='Residuals'))
fig3.add_trace(go.Scatter(x=[y_pred_exp.min(), y_pred_exp.max()], y=[0, 0], mode='lines', name='Zero Line'))
fig3.update_layout(title='Residuals vs Predicted', xaxis_title='Predicted Values', yaxis_title='Residuals')
st.plotly_chart(fig3)

# Learning curve
st.subheader('Learning Curve')
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1,
                                                        train_sizes=np.linspace(0.1, 1.0, 10))
train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean, mode='lines+markers', name='Training Error'))
fig4.add_trace(go.Scatter(x=train_sizes, y=test_scores_mean, mode='lines+markers', name='Cross-validation Error'))
fig4.update_layout(title='Learning Curve', xaxis_title='Training Set Size', yaxis_title='Mean Absolute Error')
st.plotly_chart(fig4)
