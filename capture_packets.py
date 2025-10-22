#!/usr/bin/env python3
"""
Real-Time Network Anomaly Detection Pipeline
- Works with live CSV or uploaded files
- Performs IsolationForest + LSTM anomaly detection
- Adds cross-network detection and VirusTotal enrichment
Author: Rohit Punne
"""

import os
import re
import time
import json
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from dotenv import load_dotenv

# Load VirusTotal API key from .env
load_dotenv()
VT_API = os.getenv('VIRUSTOTAL_API_KEY')

# Create data folder
os.makedirs('data', exist_ok=True)
VT_CACHE_FILE = 'data/vt_cache.json'
RELIABILITY_THRESH = 2.0

# ---------------- HELPERS ----------------
def is_private_ip(ip):
    """Check if IP belongs to a private network."""
    private_patterns = [
        r"^10\.", r"^172\.(1[6-9]|2[0-9]|3[0-1])\.", r"^192\.168\."
    ]
    return any(re.match(p, ip) for p in private_patterns)

def vt_check(ip):
    """Check IP reputation on VirusTotal with caching and rate limiting."""
    if not VT_API or not ip or not isinstance(ip, str):
        return 0

    # Load cache
    if os.path.exists(VT_CACHE_FILE):
        with open(VT_CACHE_FILE, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    if ip in cache:
        return cache[ip]

    try:
        url = f"https://www.virustotal.com/api/v3/ip_addresses/{ip}"
        headers = {"x-apikey": VT_API}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            malicious = data.get('data', {}).get('attributes', {}).get('last_analysis_stats', {}).get('malicious', 0)
            cache[ip] = malicious
            with open(VT_CACHE_FILE, "w") as f:
                json.dump(cache, f)
            time.sleep(16)  # Respect VT API rate limit
            return malicious
    except Exception:
        return 0
    return 0

# ---------------- MAIN PIPELINE ----------------
def run_pipeline(uploaded_files=None):
    """
    Run anomaly detection pipeline.
    If uploaded_files is None, uses data/realtime_network.csv.
    """
    # Load data
    if uploaded_files:
        dfs = [pd.read_csv(f) for f in uploaded_files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        INPUT_PATH = 'data/realtime_network.csv'
        if not os.path.exists(INPUT_PATH):
            raise FileNotFoundError(f"No realtime CSV found at {INPUT_PATH}")
        df = pd.read_csv(INPUT_PATH)

    # Ensure required columns
    required_cols = ['src','dst','protocol','length','ttl','flow_count','entropy_src','entropy_dst']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    # Encode protocol numerically
    le = LabelEncoder()
    df['protocol_enc'] = le.fit_transform(df['protocol'].astype(str))

    # Scale features
    features = ['length','ttl','flow_count','entropy_src','entropy_dst','protocol_enc']
    X = df[features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------- Anomaly Detection ----------------
    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['iso_anom'] = np.where(iso.fit_predict(X_scaled)==-1,1,0)

    # LSTM Autoencoder
    n_samples,n_features = X_scaled.shape
    seq = X_scaled.reshape((n_samples,1,n_features))
    auto = Sequential([
        LSTM(32,input_shape=(1,n_features),return_sequences=True),
        Dropout(0.2),
        LSTM(16,return_sequences=False),
        Dense(n_features,activation='linear')
    ])
    auto.compile(optimizer='adam',loss='mse')
    auto.fit(seq,X_scaled,epochs=6,batch_size=64,verbose=0)

    recon = auto.predict(seq)
    mse = np.mean(np.power(X_scaled - recon,2),axis=1)
    threshold = mse.mean() + RELIABILITY_THRESH * mse.std()
    df['lstm_anom'] = (mse>threshold).astype(int)

    # Ensemble & Risk Score
    df['final_alert'] = ((df['iso_anom']==1) | (df['lstm_anom']==1)).astype(int)
    df['risk_score'] = np.round((mse/threshold)*100,2)

    # Private IP & Cross-Network
    df['src_private'] = df['src'].astype(str).apply(is_private_ip)
    df['dst_private'] = df['dst'].astype(str).apply(is_private_ip)
    df['cross_network'] = ((~df['src_private']) & (df['dst_private'])).astype(int)

    # VirusTotal enrichment
    if VT_API:
        df['vt_malicious_count'] = df['src'].astype(str).apply(vt_check)
    else:
        df['vt_malicious_count'] = 0

    # Save results
    OUTPUT_PATH = 'data/final_results.csv'
    df.to_csv(OUTPUT_PATH,index=False)
    return df

# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    try:
        df_result = run_pipeline()
        print(f"[+] Pipeline completed. Results saved to data/final_results.csv")
        print(df_result.head())
    except Exception as e:
        print(f"[!] Pipeline failed: {e}")
