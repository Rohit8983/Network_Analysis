#!/usr/bin/env python3
"""
Enhanced Real-Time Network Anomaly Detection Pipeline
- Supports manual CSV uploads
- Handles multiple CSVs
- Performs preprocessing & anomaly detection (IsolationForest + LSTM Autoencoder)
- Detects private network IPs and flags cross-network traffic
- Optional VirusTotal enrichment
- Outputs in-memory results for download
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
import streamlit as st

# ---------------- CONFIGURATION ----------------
load_dotenv()
VT_API = os.getenv('VIRUSTOTAL_API_KEY')
RELIABILITY_THRESH = 2.0
VT_CACHE = {}  # in-memory cache

# ---------------- HELPERS ----------------
def is_private_ip(ip):
    """Check if IP belongs to a private network."""
    private_patterns = [
        r"^10\.", 
        r"^172\.(1[6-9]|2[0-9]|3[0-1])\.", 
        r"^192\.168\."
    ]
    return any(re.match(p, ip) for p in private_patterns)

def vt_check(ip):
    """Check IP reputation using VirusTotal with in-memory caching"""
    if not VT_API or not ip or not isinstance(ip, str):
        return 0
    if ip in VT_CACHE:
        return VT_CACHE[ip]
    try:
        url = f"https://www.virustotal.com/api/v3/ip_addresses/{ip}"
        headers = {"x-apikey": VT_API}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            malicious = r.json().get('data', {}).get('attributes', {}).get('last_analysis_stats', {}).get('malicious', 0)
            VT_CACHE[ip] = malicious
            time.sleep(16)  # respect VT API rate limit
            return malicious
    except:
        return 0
    return 0

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI Network Anomaly Detection", layout="wide")
st.title("🧠 AI Network Anomaly Detection Dashboard")

uploaded_files = st.file_uploader("Upload one or more CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    dfs = []
    for f in uploaded_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            st.warning(f"Failed to read {f.name}: {e}")
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        st.success(f"✅ Loaded {len(df)} rows from {len(dfs)} file(s)")
    else:
        st.stop()
else:
    if os.path.exists("data/realtime_network.csv"):
        df = pd.read_csv("data/realtime_network.csv")
        st.info("Using default local CSV: data/realtime_network.csv")
    else:
        st.warning("No CSV provided and no default CSV found.")
        st.stop()

# ---------------- VALIDATION ----------------
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

# ---------------- ANOMALY DETECTION ----------------
try:
    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['iso_anom'] = np.where(iso.fit_predict(X_scaled)==-1,1,0)
except Exception as e:
    st.warning(f"IsolationForest failed: {e}")
    df['iso_anom'] = 0

try:
    # LSTM Autoencoder
    n_samples, n_features = X_scaled.shape
    seq = X_scaled.reshape((n_samples,1,n_features))
    auto = Sequential([
        LSTM(32,input_shape=(1,n_features),return_sequences=True),
        Dropout(0.2),
        LSTM(16,return_sequences=False),
        Dense(n_features,activation='linear')
    ])
    auto.compile(optimizer='adam',loss='mse')
    auto.fit(seq,X_scaled,epochs=3,batch_size=64,verbose=0)
    recon = auto.predict(seq)
    mse = np.mean(np.power(X_scaled - recon,2),axis=1)
    threshold = mse.mean() + RELIABILITY_THRESH * mse.std()
    df['lstm_anom'] = (mse>threshold).astype(int)
except Exception as e:
    st.warning(f"LSTM Autoencoder failed: {e}")
    df['lstm_anom'] = 0
    mse = np.zeros(len(df))
    threshold = 1

# ---------------- RISK SCORING ----------------
df['final_alert'] = ((df['iso_anom']==1) | (df['lstm_anom']==1)).astype(int)
df['risk_score'] = np.round((mse/threshold)*100,2)

# ---------------- PRIVATE & CROSS-NETWORK ----------------
df['src_private'] = df['src'].astype(str).apply(is_private_ip)
df['dst_private'] = df['dst'].astype(str).apply(is_private_ip)
df['cross_network'] = ((~df['src_private']) & (df['dst_private'])).astype(int)

# ---------------- VIRUSTOTAL ENRICHMENT ----------------
if VT_API:
    st.info("Enriching with VirusTotal (may take time for multiple IPs)")
    df['vt_malicious_count'] = df['src'].astype(str).apply(vt_check)
else:
    df['vt_malicious_count'] = 0

# ---------------- AI SUGGESTIONS ----------------
df['ai_suggestion'] = df['risk_score'].apply(lambda x: "Monitor" if x<60 else ("Investigate" if x<85 else "Immediate Action"))

# ---------------- DASHBOARD ----------------
st.markdown("### Metrics")
col1,col2,col3,col4 = st.columns(4)
col1.metric("Total Packets", len(df))
col2.metric("Detected Alerts", int(df['final_alert'].sum()))
col3.metric("High VT Risk IPs", int((df['vt_malicious_count']>0).sum()))
col4.metric("Unique Source IPs", df['src'].nunique())

st.markdown("### Top Talkers & Destinations")
col1,col2 = st.columns(2)
col1.bar_chart(df['src'].value_counts().head(10))
col2.bar_chart(df['dst'].value_counts().head(10))

st.markdown("### High Risk Packets")
high_risk = df[df['risk_score']>80].sort_values('risk_score',ascending=False)
st.dataframe(high_risk[['src','dst','protocol','risk_score','ai_suggestion']].head(25))

# ---------------- DOWNLOAD ----------------
csv_bytes = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Processed CSV",
    data=csv_bytes,
    file_name=f"network_results_{int(time.time())}.csv",
    mime='text/csv'
)
