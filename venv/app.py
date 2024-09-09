from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load('best_random_forest_model.pkl')

@app.route('/')
def home():
    return "House Price Prediction Model"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data_df = pd.DataFrame([data])
    data_df['GrLivArea'] = np.log1p(data_df['GrLivArea'])
    data_df = pd.get_dummies(data_df).reindex(columns=X_train.columns, fill_value=0)
    prediction = model.predict(data_df)
    prediction_exp = np.expm1(prediction)
    return jsonify({'prediction': prediction_exp[0]})

if __name__ == '__main__':
    app.run(debug=True)