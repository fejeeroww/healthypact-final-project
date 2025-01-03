# app.py
from flask import Flask, request, jsonify, render_template, make_response
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load the model and scaler
try:
    model = joblib.load('demand_forecaster_model.joblib')
    scaler = joblib.load('feature_scaler.joblib')
except:
    print("Models will be loaded during prediction")

@app.route('/', methods=['GET', 'HEAD'])
def home():
    if request.method == 'HEAD':
        return make_response('', 200)
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
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
            
            # Scale features and predict
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)[0]
            
            return render_template('predict.html', 
                                 prediction=round(prediction, 2),
                                 success=True)
        except Exception as e:
            return render_template('predict.html', 
                                 error=str(e),
                                 success=False)
    
    return render_template('predict.html')

@app.route('/optimize', methods=['GET', 'POST'])
def optimize():
    if request.method == 'POST':
        try:
            # Get form data
            date = datetime.strptime(request.form['date'], '%Y-%m-%d')
            promotion = int(request.form['promotion'])
            economic_index = float(request.form['economic_index'])
            competitor_price = float(request.form['competitor_price'])
            
            # Calculate time features
            month = date.month
            is_weekend = 1 if date.weekday() >= 5 else 0
            month_sin = np.sin(2 * np.pi * month/12)
            month_cos = np.cos(2 * np.pi * month/12)
            
            # Price optimization
            price_range = np.arange(10, 100, 1)
            best_price = 0
            max_revenue = 0
            demand_at_optimal = 0
            
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
                    demand_at_optimal = predicted_demand
            
            return render_template('optimize.html', 
                                 optimal_price=round(best_price, 2),
                                 expected_revenue=round(max_revenue, 2),
                                 optimal_demand=round(demand_at_optimal, 2),
                                 success=True)
        except Exception as e:
            return render_template('optimize.html', 
                                 error=str(e),
                                 success=False)
    
    return render_template('optimize.html')

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True)
