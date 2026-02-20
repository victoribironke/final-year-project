"""
Flask backend for the Demand Forecasting Prediction UI.

Trains models from the harmonized CSV on first boot (if no saved models exist),
then serves a prediction API + a stunning frontend.
"""
import os
import json
import math
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

app = Flask(__name__)

# ── Paths ──────────────────────────────────────────────────
CSV_PATH = "harmonized_food_prices.csv"
MODEL_DIR = "models"
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

# ── Feature config (must match the notebook) ──────────────
FEATURE_COLS = [
    'Demand_Lag1', 'Demand_Lag3', 'Demand_Lag6', 'Demand_Lag12',
    'Demand_RollingMean3', 'Demand_RollingMean6', 'Demand_RollingMean12',
    'Avg_Temperature', 'Rainfall', 'Is_Holiday',
    'Month_Sin', 'Month_Cos', 'Is_Wet_Season', 'Year'
]
TARGET_COL = 'Demand'
MIN_SAMPLES = 36

# ── Feature engineering ───────────────────────────────────
def create_features(group_df):
    df = group_df.copy()
    df['Demand_Lag1']  = df['Demand'].shift(1)
    df['Demand_Lag3']  = df['Demand'].shift(3)
    df['Demand_Lag6']  = df['Demand'].shift(6)
    df['Demand_Lag12'] = df['Demand'].shift(12)
    df['Demand_RollingMean3']  = df['Demand'].rolling(window=3).mean()
    df['Demand_RollingMean6']  = df['Demand'].rolling(window=6).mean()
    df['Demand_RollingMean12'] = df['Demand'].rolling(window=12).mean()
    df['Month'] = df['Date'].dt.month
    df['Year']  = df['Date'].dt.year
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Is_Wet_Season'] = df['Month'].apply(lambda m: 1 if 4 <= m <= 10 else 0)
    return df


# ── Globals filled at startup ─────────────────────────────
commodity_models = {}   # {commodity: {'model': ..., 'scaler_X': ..., 'scaler_y': ..., 'champion': ...}}
commodity_list = []
df_raw = None
commodity_summaries = {}  # per‑commodity stats for the frontend


def _train_models_from_csv():
    """Train a Random Forest for every commodity and cache results."""
    global df_raw
    print("[BOOT] Loading CSV …")
    df_raw = pd.read_csv(CSV_PATH)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    
    # Aggregate
    df_agg = df_raw.groupby(['Date', 'Commodity_Name']).agg({
        'Demand': 'mean',
        'Unit_Price': 'mean',
        'Avg_Temperature': 'mean',
        'Rainfall': 'mean',
        'Is_Holiday': 'max',
    }).reset_index().sort_values(['Commodity_Name', 'Date']).reset_index(drop=True)
    df_agg['Avg_Temperature'] = df_agg.groupby('Commodity_Name')['Avg_Temperature'].transform(lambda x: x.ffill().bfill())
    df_agg['Rainfall'] = df_agg.groupby('Commodity_Name')['Rainfall'].transform(lambda x: x.ffill().bfill())

    # Feature engineering
    print("[BOOT] Engineering features …")
    dfs = []
    for commodity, grp in df_agg.groupby('Commodity_Name'):
        dfs.append(create_features(grp))
    df_features = pd.concat(dfs, ignore_index=True)
    
    valid_commodities = [
        c for c, grp in df_features.groupby('Commodity_Name')
        if grp.dropna(subset=FEATURE_COLS + [TARGET_COL]).shape[0] >= MIN_SAMPLES
    ]
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    for commodity in valid_commodities:
        crop = df_features[df_features['Commodity_Name'] == commodity].dropna(subset=FEATURE_COLS + [TARGET_COL]).sort_values('Date').reset_index(drop=True)
        split = int(len(crop) * 0.8)
        X_train = crop[FEATURE_COLS].values[:split]
        y_train = crop[TARGET_COL].values[:split]
        
        # Random Forest (fast and reliable)
        rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        safe = commodity.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        joblib.dump(rf, os.path.join(MODEL_DIR, f"{safe}_model.pkl"))
        
        commodity_models[commodity] = {
            'model': rf,
            'champion': 'Random Forest',
            'feature_cols': FEATURE_COLS,
        }
        
        # Store summary stats for the frontend
        commodity_summaries[commodity] = {
            'records': len(crop),
            'date_range': [crop['Date'].min().strftime('%Y-%m'), crop['Date'].max().strftime('%Y-%m')],
            'avg_demand': round(float(crop['Demand'].mean()), 2),
            'avg_temp': round(float(crop['Avg_Temperature'].mean()), 2) if not crop['Avg_Temperature'].isna().all() else None,
            'avg_rainfall': round(float(crop['Rainfall'].mean()), 2) if not crop['Rainfall'].isna().all() else None,
        }
        
    print(f"[BOOT] Trained {len(commodity_models)} commodity models")
    return df_features


def _load_saved_models():
    """Load models saved by the notebook (if available)."""
    global df_raw
    if not os.path.exists(METADATA_PATH):
        return False
    
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    
    for commodity, info in meta.get('commodity_stats', {}).items():
        safe = commodity.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        model_path = os.path.join(MODEL_DIR, f"{safe}_model.pkl")
        if os.path.exists(model_path):
            commodity_models[commodity] = {
                'model': joblib.load(model_path),
                'champion': info.get('champion_model', 'Unknown'),
                'feature_cols': meta.get('feature_cols', FEATURE_COLS),
            }
    
    if commodity_models:
        # Load CSV for summaries
        df_raw = pd.read_csv(CSV_PATH)
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
        df_agg = df_raw.groupby(['Date', 'Commodity_Name']).agg({
            'Demand': 'mean',
            'Avg_Temperature': 'mean',
            'Rainfall': 'mean',
        }).reset_index()
        
        for commodity in commodity_models:
            crop = df_agg[df_agg['Commodity_Name'] == commodity]
            commodity_summaries[commodity] = {
                'records': len(crop),
                'date_range': [crop['Date'].min().strftime('%Y-%m'), crop['Date'].max().strftime('%Y-%m')],
                'avg_demand': round(float(crop['Demand'].mean()), 2),
                'avg_temp': round(float(crop['Avg_Temperature'].mean()), 2) if not crop['Avg_Temperature'].isna().all() else None,
                'avg_rainfall': round(float(crop['Rainfall'].mean()), 2) if not crop['Rainfall'].isna().all() else None,
            }
        
        print(f"[BOOT] Loaded {len(commodity_models)} saved models")
        return True
    return False


def boot():
    """Initialize models on startup."""
    global commodity_list
    if not _load_saved_models():
        _train_models_from_csv()
    commodity_list = sorted(commodity_models.keys())


# ── Prediction logic ──────────────────────────────────────
def predict_demand(commodity, temperature, rainfall, month, year, is_holiday,
                   recent_demands=None):
    """
    Predict demand for a commodity given input features.
    recent_demands: list of up to 12 recent demand values (most recent first)
    """
    if commodity not in commodity_models:
        return None, "Commodity not found"
    
    info = commodity_models[commodity]
    model = info['model']
    
    # Build features
    if recent_demands is None or len(recent_demands) == 0:
        # Use the commodity's average demand as a baseline
        avg = commodity_summaries.get(commodity, {}).get('avg_demand', 50.0)
        recent_demands = [avg] * 12
    else:
        # Pad to 12 if shorter
        while len(recent_demands) < 12:
            recent_demands.append(recent_demands[-1])
    
    # Lag features
    lag1  = recent_demands[0]
    lag3  = recent_demands[2] if len(recent_demands) > 2 else recent_demands[0]
    lag6  = recent_demands[5] if len(recent_demands) > 5 else recent_demands[0]
    lag12 = recent_demands[11] if len(recent_demands) > 11 else recent_demands[0]
    
    # Rolling means
    rm3  = np.mean(recent_demands[:3])
    rm6  = np.mean(recent_demands[:6])
    rm12 = np.mean(recent_demands[:12])
    
    # Cyclical month encoding
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Season
    is_wet = 1 if 4 <= month <= 10 else 0
    
    features = np.array([[
        lag1, lag3, lag6, lag12,
        rm3, rm6, rm12,
        temperature, rainfall, int(is_holiday),
        month_sin, month_cos, is_wet, year
    ]])
    
    # Handle SVR with scalers
    if isinstance(model, dict) and 'model' in model:
        svr = model['model']
        scaler_X = model['scaler_X']
        scaler_y = model['scaler_y']
        features_scaled = scaler_X.transform(features)
        pred_scaled = svr.predict(features_scaled)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
    else:
        pred = float(model.predict(features)[0])
    
    return round(pred, 2), None


# ── Routes ────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/commodities')
def api_commodities():
    """Return list of commodities + summaries."""
    data = []
    for c in commodity_list:
        item = {'name': c, 'champion': commodity_models[c]['champion']}
        item.update(commodity_summaries.get(c, {}))
        data.append(item)
    return jsonify(data)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Predict demand for a commodity."""
    body = request.json
    commodity = body.get('commodity')
    temperature = float(body.get('temperature', 28))
    rainfall = float(body.get('rainfall', 100))
    month = int(body.get('month', datetime.now().month))
    year = int(body.get('year', datetime.now().year))
    is_holiday = bool(body.get('is_holiday', False))
    recent_demands = body.get('recent_demands', None)
    
    pred, err = predict_demand(commodity, temperature, rainfall, month, year, is_holiday, recent_demands)
    if err:
        return jsonify({'error': err}), 400
    
    return jsonify({
        'commodity': commodity,
        'predicted_demand': pred,
        'champion_model': commodity_models[commodity]['champion'],
        'inputs': {
            'temperature': temperature,
            'rainfall': rainfall,
            'month': month,
            'year': year,
            'is_holiday': is_holiday,
        }
    })


@app.route('/api/history/<commodity>')
def api_history(commodity):
    """Return demand history for a commodity (aggregated monthly)."""
    if df_raw is None:
        return jsonify([])
    
    crop = df_raw[df_raw['Commodity_Name'] == commodity].copy()
    crop['YearMonth'] = crop['Date'].dt.to_period('M').astype(str)
    monthly = crop.groupby('YearMonth').agg({
        'Demand': 'mean',
        'Unit_Price': 'mean',
        'Avg_Temperature': 'mean',
        'Rainfall': 'mean',
    }).reset_index()
    
    result = []
    for _, row in monthly.iterrows():
        result.append({
            'date': row['YearMonth'],
            'demand': round(float(row['Demand']), 2) if not pd.isna(row['Demand']) else None,
            'price': round(float(row['Unit_Price']), 2) if not pd.isna(row['Unit_Price']) else None,
            'temperature': round(float(row['Avg_Temperature']), 2) if not pd.isna(row['Avg_Temperature']) else None,
            'rainfall': round(float(row['Rainfall']), 2) if not pd.isna(row['Rainfall']) else None,
        })
    
    return jsonify(result)


if __name__ == '__main__':
    boot()
    print(f"\n{'=' * 50}")
    print(f"  Demand Forecasting UI")
    print(f"  http://127.0.0.1:5000")
    print(f"  {len(commodity_list)} commodities loaded")
    print(f"{'=' * 50}\n")
    app.run(debug=False, port=5000)
