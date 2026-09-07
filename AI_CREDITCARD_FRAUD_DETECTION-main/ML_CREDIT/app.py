from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import pickle
import json
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    """Serve the frontend HTML page"""
    return send_file(os.path.join(BASE_DIR, 'nexmart_final.html'))

@app.route('/images/<path:filename>')
def serve_images(filename):
    """Serve image files"""
    return send_from_directory(os.path.join(BASE_DIR, 'images'), filename)

# Load model and metadata
with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_meta.json', 'r') as f:
    meta = json.load(f)

FEATURES = meta['features']
cat_fraud_rate = meta['category_fraud_rate']

# Session storage per card
last_transaction = {}
card_session = {}

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates in kilometers"""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

@app.route('/score', methods=['POST'])
def score():
    data = request.json
    
    # Extract transaction data
    cc_num = data['cc_num']
    amt = float(data['amt'])
    category = data['category']
    merch_lat = float(data['merch_lat'])
    merch_long = float(data['merch_long'])
    user_lat = float(data['user_lat'])
    user_long = float(data['user_long'])
    unix_time = float(data['unix_time'])
    
    now = datetime.fromtimestamp(unix_time)
    
    # IMPOSSIBLE TRAVELER DETECTION
    prev = last_transaction.get(cc_num)
    if prev:
        consec_dist = haversine(prev['merch_lat'], prev['merch_long'], merch_lat, merch_long)
        time_gap = max((unix_time - prev['unix_time']) / 3600, 0.0001)
        travel_speed = consec_dist / time_gap
    else:
        consec_dist = 0
        time_gap = 24
        travel_speed = 0
    
    is_impossible = int(travel_speed > 500)
    is_fast = int(travel_speed > 200)
    
    # HOME TO MERCHANT DISTANCE
    dist_home = haversine(user_lat, user_long, merch_lat, merch_long)
    
    # TIME FEATURES
    hour = now.hour
    dow = now.weekday()
    month = now.month
    is_night = int(hour >= 22 or hour <= 5)
    is_weekend = int(dow >= 5)
    
    # VELOCITY TRACKING
    session = card_session.get(cc_num, {'tx_times': [], 'tx_amounts': []})
    session['tx_times'].append(unix_time)
    session['tx_amounts'].append(amt)
    card_session[cc_num] = session
    
    one_hour_ago = unix_time - 3600
    one_day_ago = unix_time - 86400
    tx_count_1h = sum(1 for t in session['tx_times'] if t >= one_hour_ago)
    tx_count_1d = sum(1 for t in session['tx_times'] if t >= one_day_ago)
    
    # AMOUNT CHANGE
    if len(session['tx_amounts']) > 1:
        prev_amt = session['tx_amounts'][-2]
        amt_change = amt / max(prev_amt, 1)
    else:
        amt_change = 1.0
    
    # BEHAVIORAL PROFILE
    card_avg = float(data.get('card_avg_amt', amt))
    card_std = float(data.get('card_std_amt', 1))
    
    if len(session['tx_amounts']) >= 3:
        card_avg = float(np.mean(session['tx_amounts']))
        card_std = float(np.std(session['tx_amounts'])) or 1.0
    
    amt_zscore = (amt - card_avg) / max(card_std, 1)
    is_anomaly = int(amt_zscore > 3)
    
    # CATEGORY FRAUD RATE
    cat_rate = cat_fraud_rate.get(category, 0.005)
    
    # ENCODE CATEGORICALS
    gender_enc = 0 if data.get('gender', 'M') == 'F' else 1
    state_enc = hash(data.get('state', 'CA')) % 50
    category_enc = hash(category) % 14
    
    # BUILD FEATURE VECTOR
    features = {
        'amt': amt,
        'hour': hour,
        'day_of_week': dow,
        'month': month,
        'is_night': is_night,
        'is_weekend': is_weekend,
        'age': int(data.get('age', 35)),
        'gender_enc': gender_enc,
        'city_pop': int(data.get('city_pop', 50000)),
        'dist_home_to_merchant': dist_home,
        'consec_dist_km': consec_dist,
        'time_gap_hours': time_gap,
        'travel_speed_kmh': travel_speed,
        'is_impossible_traveler': is_impossible,
        'is_fast_traveler': is_fast,
        'tx_count_1h': tx_count_1h,
        'tx_count_1d': tx_count_1d,
        'amt_change_ratio': amt_change,
        'amt_zscore': amt_zscore,
        'card_avg_amt': card_avg,
        'card_std_amt': card_std,
        'is_amt_anomaly': is_anomaly,
        'category_fraud_rate': cat_rate,
        'state_enc': state_enc,
        'category_enc': category_enc,
    }
    
    # PREDICT FRAUD
    X = pd.DataFrame([features])[FEATURES]
    fraud_prob = model.predict_proba(X)[0][1]
    is_fraud = bool(fraud_prob >= 0.5)
    
    # SAVE FOR NEXT TRANSACTION
    last_transaction[cc_num] = {
        'merch_lat': merch_lat,
        'merch_long': merch_long,
        'unix_time': unix_time,
        'amt': amt,
    }
    
    # BUILD FLAGS
    flags = []
    if is_impossible:
        flags.append("IMPOSSIBLE TRAVELER: Physical travel speed exceeds 500 km/h")
    if is_fast and not is_impossible:
        flags.append("SUSPICIOUS SPEED: Travel speed over 200 km/h detected")
    if is_anomaly:
        flags.append("UNUSUAL AMOUNT: Spend deviates 3+ std from card baseline")
    if tx_count_1h >= 5:
        flags.append(f"EXTREME VELOCITY: {tx_count_1h} transactions in 60 minutes")
    elif tx_count_1h >= 3:
        flags.append(f"HIGH VELOCITY: {tx_count_1h} transactions in 1 hour")
    if amt_change > 5:
        flags.append(f"AMOUNT SPIKE: {amt_change:.1f}x increase from previous transaction")
    
    return jsonify({
        'fraud_probability': round(float(fraud_prob) * 100, 2),
        'is_fraud': is_fraud,
        'flags': flags,
        'details': {
            'travel_speed_kmh': round(travel_speed, 1),
            'consec_dist_km': round(consec_dist, 1),
            'time_gap_hours': round(time_gap, 2),
            'amt_zscore': round(amt_zscore, 2),
            'dist_home_to_merchant': round(dist_home, 1),
            'tx_count_1h': tx_count_1h,
            'tx_count_1d': tx_count_1d,
            'card_avg_amt': round(card_avg, 2),
            'card_std_amt': round(card_std, 2),
            'amt_change_ratio': round(amt_change, 2),
        }
    })

@app.route('/reset', methods=['POST'])
def reset():
    """Clear all session data"""
    last_transaction.clear()
    card_session.clear()
    return jsonify({'status': 'success', 'message': 'All session data cleared'})

@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint"""
    return jsonify({'status': 'online', 'model_loaded': True})

if __name__ == '__main__':
    print("=" * 60)
    print("FRAUD DETECTION API SERVER")
    print("=" * 60)
    print("Status: Running")
    print("Address: http://localhost:5000")
    print("Model: GradientBoosting (AUC-ROC: 0.9849)")
    print("=" * 60)
    app.run(debug=True, port=5000, host='127.0.0.1')
