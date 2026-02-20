"""
Flask backend for the Demand Forecasting Prediction UI.

Optimised for low-memory deployment:
  - Models are loaded lazily (only when a prediction is requested)
  - CSV is aggregated once at startup but kept small
  - Works with or without pre-trained models from the notebook
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

# ── Globals (lightweight — no models loaded at startup) ───
commodity_list = []
commodity_summaries = {}   # {commodity: {records, date_range, avg_demand, ...}}
commodity_champions = {}   # {commodity: champion_model_name}
_model_cache = {}          # Lazy cache: {commodity: loaded_model}
_df_features = None        # Cached feature-engineered DataFrame (only if no models/ dir)


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


def _safe_name(commodity):
    return commodity.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')


# ── Boot: lightweight index only ──────────────────────────
def boot():
    """Build commodity index and summaries. No models loaded yet."""
    global commodity_list, _df_features

    has_saved_models = os.path.exists(METADATA_PATH)

    # ── Load CSV and build summaries ──────────────────────
    print("[BOOT] Loading CSV ...")
    df_raw = pd.read_csv(CSV_PATH)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])

    df_agg = df_raw.groupby(['Date', 'Commodity_Name']).agg({
        'Demand': 'mean',
        'Unit_Price': 'mean',
        'Avg_Temperature': 'mean',
        'Rainfall': 'mean',
        'Is_Holiday': 'max',
    }).reset_index().sort_values(['Commodity_Name', 'Date']).reset_index(drop=True)

    df_agg['Avg_Temperature'] = df_agg.groupby('Commodity_Name')['Avg_Temperature'].transform(lambda x: x.ffill().bfill())
    df_agg['Rainfall'] = df_agg.groupby('Commodity_Name')['Rainfall'].transform(lambda x: x.ffill().bfill())

    # Build summaries for every commodity
    for commodity, crop in df_agg.groupby('Commodity_Name'):
        commodity_summaries[commodity] = {
            'records': len(crop),
            'date_range': [crop['Date'].min().strftime('%Y-%m'), crop['Date'].max().strftime('%Y-%m')],
            'avg_demand': round(float(crop['Demand'].mean()), 2),
            'avg_temp': round(float(crop['Avg_Temperature'].mean()), 2) if not crop['Avg_Temperature'].isna().all() else None,
            'avg_rainfall': round(float(crop['Rainfall'].mean()), 2) if not crop['Rainfall'].isna().all() else None,
        }

    if has_saved_models:
        # ── Use saved models: just read metadata ──────────
        print("[BOOT] Found models/ directory, using saved models (lazy loading)")
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        for commodity, info in meta.get('commodity_stats', {}).items():
            commodity_champions[commodity] = info.get('champion_model', 'Unknown')
        commodity_list = sorted(commodity_champions.keys())
    else:
        # ── No saved models: prepare data for on-demand training ──
        print("[BOOT] No models/ directory, will train on-demand")
        print("[BOOT] Engineering features ...")
        dfs = []
        for commodity, grp in df_agg.groupby('Commodity_Name'):
            dfs.append(create_features(grp))
        _df_features = pd.concat(dfs, ignore_index=True)

        valid = [
            c for c, grp in _df_features.groupby('Commodity_Name')
            if grp.dropna(subset=FEATURE_COLS + [TARGET_COL]).shape[0] >= MIN_SAMPLES
        ]
        for c in valid:
            commodity_champions[c] = 'Random Forest'
        commodity_list = sorted(valid)

    # Free raw DataFrame
    del df_raw
    print(f"[BOOT] Ready: {len(commodity_list)} commodities indexed")


# ── Lazy model loading ────────────────────────────────────
def _train_rf_from_csv(commodity):
    """Train a quick Random Forest for a commodity from the CSV."""
    print(f"  [TRAIN] Training fallback RF for {commodity} ...")
    df_raw = pd.read_csv(CSV_PATH)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])

    df_agg = df_raw.groupby(['Date', 'Commodity_Name']).agg({
        'Demand': 'mean', 'Avg_Temperature': 'mean',
        'Rainfall': 'mean', 'Is_Holiday': 'max',
    }).reset_index().sort_values(['Commodity_Name', 'Date']).reset_index(drop=True)

    df_agg['Avg_Temperature'] = df_agg.groupby('Commodity_Name')['Avg_Temperature'].transform(lambda x: x.ffill().bfill())
    df_agg['Rainfall'] = df_agg.groupby('Commodity_Name')['Rainfall'].transform(lambda x: x.ffill().bfill())

    crop_raw = df_agg[df_agg['Commodity_Name'] == commodity].copy()
    if crop_raw.empty:
        return None

    crop = create_features(crop_raw).dropna(subset=FEATURE_COLS + [TARGET_COL]).sort_values('Date').reset_index(drop=True)
    if len(crop) < MIN_SAMPLES:
        return None

    split = int(len(crop) * 0.8)
    X_train = crop[FEATURE_COLS].values[:split]
    y_train = crop[TARGET_COL].values[:split]

    rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                               min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Save as .pkl so it loads instantly next time
    safe = _safe_name(commodity)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf, os.path.join(MODEL_DIR, f"{safe}_model.pkl"))

    # Update the champion label
    commodity_champions[commodity] = 'Random Forest (fallback)'
    print(f"  [TRAIN] Done — saved as {safe}_model.pkl")
    del df_raw
    return rf


def _get_model(commodity):
    """Load or train a model for the commodity, caching the result."""
    if commodity in _model_cache:
        return _model_cache[commodity]

    safe = _safe_name(commodity)
    pkl_path = os.path.join(MODEL_DIR, f"{safe}_model.pkl")
    keras_path = os.path.join(MODEL_DIR, f"{safe}_model.keras")

    # 1) Try .pkl (sklearn / xgboost / SVR)
    if os.path.exists(pkl_path):
        model = joblib.load(pkl_path)
        _model_cache[commodity] = model
        return model

    # 2) Try .keras (LSTM / GRU) — requires TensorFlow
    if os.path.exists(keras_path):
        try:
            from tensorflow.keras.models import load_model
            model = load_model(keras_path)
            _model_cache[commodity] = model
            return model
        except ImportError:
            print(f"  [WARN] {commodity}: .keras model found but TensorFlow not installed, training fallback RF ...")
        except Exception as e:
            print(f"  [WARN] {commodity}: failed to load .keras model ({e}), training fallback RF ...")

    # 3) Fallback: train from _df_features if available
    if _df_features is not None:
        crop = _df_features[_df_features['Commodity_Name'] == commodity].dropna(
            subset=FEATURE_COLS + [TARGET_COL]
        ).sort_values('Date').reset_index(drop=True)

        if len(crop) >= MIN_SAMPLES:
            split = int(len(crop) * 0.8)
            X_train = crop[FEATURE_COLS].values[:split]
            y_train = crop[TARGET_COL].values[:split]

            rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                                       min_samples_split=5, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)

            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(rf, pkl_path)
            commodity_champions[commodity] = 'Random Forest (fallback)'
            _model_cache[commodity] = rf
            return rf

    # 4) Last resort: train from CSV directly
    rf = _train_rf_from_csv(commodity)
    if rf:
        _model_cache[commodity] = rf
        return rf

    return None


# ── Prediction logic ──────────────────────────────────────
def predict_demand(commodity, temperature, rainfall, month, year, is_holiday,
                   recent_demands=None):
    if commodity not in commodity_champions:
        return None, "Commodity not found"

    model = _get_model(commodity)
    if model is None:
        return None, "Could not load model"

    # Build feature vector
    if recent_demands is None or len(recent_demands) == 0:
        avg = commodity_summaries.get(commodity, {}).get('avg_demand', 50.0)
        recent_demands = [avg] * 12
    else:
        while len(recent_demands) < 12:
            recent_demands.append(recent_demands[-1])

    lag1  = recent_demands[0]
    lag3  = recent_demands[2] if len(recent_demands) > 2 else recent_demands[0]
    lag6  = recent_demands[5] if len(recent_demands) > 5 else recent_demands[0]
    lag12 = recent_demands[11] if len(recent_demands) > 11 else recent_demands[0]

    rm3  = np.mean(recent_demands[:3])
    rm6  = np.mean(recent_demands[:6])
    rm12 = np.mean(recent_demands[:12])

    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    is_wet = 1 if 4 <= month <= 10 else 0

    features = np.array([[
        lag1, lag3, lag6, lag12,
        rm3, rm6, rm12,
        temperature, rainfall, int(is_holiday),
        month_sin, month_cos, is_wet, year
    ]])

    # Handle SVR dict format
    if isinstance(model, dict) and 'model' in model:
        svr = model['model']
        features_scaled = model['scaler_X'].transform(features)
        pred_scaled = svr.predict(features_scaled)
        pred = model['scaler_y'].inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
    else:
        pred = float(model.predict(features)[0])

    return round(pred, 2), None


# ── Routes ────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/commodities')
def api_commodities():
    data = []
    for c in commodity_list:
        item = {'name': c, 'champion': commodity_champions.get(c, 'Random Forest')}
        item.update(commodity_summaries.get(c, {}))
        data.append(item)
    return jsonify(data)


@app.route('/api/predict', methods=['POST'])
def api_predict():
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
        'champion_model': commodity_champions.get(commodity, 'Random Forest'),
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
    """Return demand history (reads CSV on-demand to save memory)."""
    try:
        df = pd.read_csv(CSV_PATH, usecols=['Date', 'Commodity_Name', 'Demand', 'Unit_Price', 'Avg_Temperature', 'Rainfall'])
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception:
        return jsonify([])

    crop = df[df['Commodity_Name'] == commodity].copy()
    crop['YearMonth'] = crop['Date'].dt.to_period('M').astype(str)
    monthly = crop.groupby('YearMonth').agg({
        'Demand': 'mean', 'Unit_Price': 'mean',
        'Avg_Temperature': 'mean', 'Rainfall': 'mean',
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

    del df  # Free memory
    return jsonify(result)


if __name__ == '__main__':
    boot()
    print(f"\n{'=' * 50}")
    print(f"  Demand Forecasting UI")
    print(f"  http://127.0.0.1:5000")
    print(f"  {len(commodity_list)} commodities ready")
    print(f"{'=' * 50}\n")
    app.run(debug=False, port=5000)
