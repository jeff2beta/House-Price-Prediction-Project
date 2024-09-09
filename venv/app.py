import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load('best_random_forest_model.pkl')

# Load training data columns (ensure this is consistent with your model's expectations)
X_train = pd.read_csv('house-prices/train.csv')
X_train = pd.get_dummies(X_train.drop(columns=['SalePrice']))

# Streamlit App Title
st.title('House Price Prediction')

# Function to get user input features
def user_input_features():
    GrLivArea = st.number_input('Above Ground Living Area (square feet)', min_value=500, max_value=8000, value=1500)
    OverallQual = st.slider('Overall Quality', 1, 10, 5)
    YearBuilt = st.slider('Year Built', 1900, 2020, 2000)
    GarageCars = st.slider('Number of Garage Cars', 0, 4, 2)

    # Collect user input into a dataframe
    data = {
        'GrLivArea': GrLivArea,
        'OverallQual': OverallQual,
        'YearBuilt': YearBuilt,
        'GarageCars': GarageCars
    }

    features = pd.DataFrame([data])
    return features

# Get user input
input_df = user_input_features()

# Preprocess the input (same way as during training)
input_df['GrLivArea'] = np.log1p(input_df['GrLivArea'])
input_df = pd.get_dummies(input_df).reindex(columns=X_train.columns, fill_value=0)

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_exp = np.expm1(prediction)
    
    # Display the prediction result
    st.subheader('Predicted House Price')
    st.write(f'${prediction_exp[0]:,.2f}')