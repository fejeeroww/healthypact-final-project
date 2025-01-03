# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('demand_forecaster_model.joblib')
scaler = joblib.load('feature_scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get values from the form
        date = datetime.strptime(request.form['date'], '%Y-%m-%d')
        price = float(request.form['price'])
        promotion = int(request.form['promotion'])
        economic_index = float(request.form['economic_index'])
        competitor_price = float(request.form['competitor_price'])
        
        # Calculate time-based features
        month = date.month
        is_weekend = 1 if date.weekday() >= 5 else 0
        month_sin = np.sin(2 * np.pi * month/12)
        month_cos = np.cos(2 * np.pi * month/12)
        
        # Create feature array
        features = np.array([
            month_sin,
            month_cos,
            is_weekend,
            price,
            promotion,
            economic_index,
            competitor_price,
            1400,  # default demand_lag_7
            1380,  # default demand_lag_14
            1420   # default rolling_mean_7
        ]).reshape(1, -1)
        
        # Scale features
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        return render_template('predict.html', prediction=round(prediction, 2))
    
    return render_template('predict.html')

@app.route('/optimize', methods=['GET', 'POST'])
def optimize():
    if request.method == 'POST':
        date = datetime.strptime(request.form['date'], '%Y-%m-%d')
        promotion = int(request.form['promotion'])
        economic_index = float(request.form['economic_index'])
        competitor_price = float(request.form['competitor_price'])
        
        month = date.month
        is_weekend = 1 if date.weekday() >= 5 else 0
        month_sin = np.sin(2 * np.pi * month/12)
        month_cos = np.cos(2 * np.pi * month/12)
        
        # Price optimization
        price_range = np.arange(10, 100, 1)
        best_price = 0
        max_revenue = 0
        
        for price in price_range:
            features = np.array([
                month_sin,
                month_cos,
                is_weekend,
                price,
                promotion,
                economic_index,
                competitor_price,
                1400,  # default demand_lag_7
                1380,  # default demand_lag_14
                1420   # default rolling_mean_7
            ]).reshape(1, -1)
            
            scaled_features = scaler.transform(features)
            predicted_demand = model.predict(scaled_features)[0]
            revenue = predicted_demand * price
            
            if revenue > max_revenue:
                max_revenue = revenue
                best_price = price
        
        return render_template('optimize.html', 
                             optimal_price=round(best_price, 2),
                             expected_revenue=round(max_revenue, 2))
    
    return render_template('optimize.html')

if __name__ == '__main__':
    app.run(debug=True)
