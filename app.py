#!/usr/bin/env python3
"""
Optimized AI-Powered Network Dashboard
Author: Rohit Punne
Features: Real-time analytics, IP geolocation (cached), AI anomaly detection, faster visualizations, fast mode toggle
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os, glob, time, socket, paramiko, requests, json
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import google.generativeai as genai
import textwrap


# --- Config ---
st.set_page_config(page_title='Advanced AI Network Dashboard', layout='wide', page_icon="🧠")
st.title("🧠 Optimized AI-Powered Network Dashboard")
REFRESH_SECONDS = 8
DATA_FOLDER = "data"
GEO_CACHE_FILE = os.path.join(DATA_FOLDER, "ip_geo_cache.json")

# --- Required Columns ---
REQUIRED_COLS = [
    'timestamp','src','dst','protocol','length','ttl','flow_count',
    'entropy_src','entropy_dst','iso_anom','lstm_anom','final_alert',
    'risk_score','cross_network','vt_malicious_count'
]

PRIVATE_CONFIG = {
    "host": "172.16.0.254",
    "port": 22,
    "username": "239",
    "password": "your_password",
    "remote_path": "/home/data/live_traffic.csv"
}

# --- Helper Functions ---
def get_latest_csv(folder=DATA_FOLDER):
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    if not csv_files: return None
    return max(csv_files, key=os.path.getmtime)

def check_private_network(host, port=22):
    try:
        socket.setdefaulttimeout(3)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.close()
        return True
    except:
        return False

def geolocate_ip(ip, geo_cache):
    if ip in geo_cache:
        return geo_cache[ip]
    try:
        response = requests.get(f"https://ipapi.co/{ip}/json/", timeout=3)
        if response.status_code == 200:
            data = response.json()
            geo_cache[ip] = (data.get("city","Unknown"), data.get("country_name","Unknown"))
        else:
            geo_cache[ip] = ("Unknown","Unknown")
    except:
        geo_cache[ip] = ("Unknown","Unknown")
    return geo_cache[ip]

# Load geo cache
if os.path.exists(GEO_CACHE_FILE):
    with open(GEO_CACHE_FILE, "r") as f:
        geo_cache = json.load(f)
else:
    geo_cache = {}

# --- Sidebar: Data Source & Fast Mode ---
st.sidebar.header("Data Source Configuration")
uploaded_file = st.sidebar.file_uploader("📤 Upload CSV file", type="csv")
auto_mode = st.sidebar.checkbox("🔍 Auto-detect Private Network", value=True)
fast_mode = st.sidebar.checkbox("⚡ Fast Mode (Skip LSTM & Geolocation)", value=False)

if uploaded_file:
    data_source = "Uploaded CSV"
elif auto_mode:
    connected_private = check_private_network(PRIVATE_CONFIG["host"], PRIVATE_CONFIG["port"])
    if connected_private:
        data_source = "SFTP/Private Network"
        st.sidebar.success(f"✅ Connected to Private Network ({PRIVATE_CONFIG['host']})")
    else:
        data_source = "Local CSV"
        st.sidebar.warning("⚠️ Private network unreachable — using latest local CSV.")
else:
    data_source = st.sidebar.radio("Choose Data Source:", ["Local CSV", "SFTP/Private Network"])

# Manual SFTP Config
if not auto_mode and data_source=="SFTP/Private Network":
    PRIVATE_CONFIG["host"] = st.sidebar.text_input("SFTP Host", PRIVATE_CONFIG["host"])
    PRIVATE_CONFIG["port"] = st.sidebar.number_input("Port", 22)
    PRIVATE_CONFIG["username"] = st.sidebar.text_input("Username", PRIVATE_CONFIG["username"])
    PRIVATE_CONFIG["password"] = st.sidebar.text_input("Password", type="password")
    PRIVATE_CONFIG["remote_path"] = st.sidebar.text_input("Remote CSV Path", PRIVATE_CONFIG["remote_path"])
elif data_source=="Local CSV":
    LOCAL_PATH = get_latest_csv()
    if LOCAL_PATH:
        st.sidebar.info(f"📁 Using latest CSV: {os.path.basename(LOCAL_PATH)}")
    else:
        st.sidebar.warning("⚠️ No CSV found in data folder")

# Auto Refresh
st_autorefresh(interval=REFRESH_SECONDS*1000, key="data_refresh")

# --- Load Data with caching ---
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

df = pd.DataFrame()
try:
    if data_source=="SFTP/Private Network":
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=PRIVATE_CONFIG["host"],
            port=PRIVATE_CONFIG["port"],
            username=PRIVATE_CONFIG["username"],
            password=PRIVATE_CONFIG["password"],
            timeout=5
        )
        sftp = client.open_sftp()
        with sftp.open(PRIVATE_CONFIG["remote_path"], "r") as f:
            df = pd.read_csv(f)
        sftp.close()
        client.close()
        st.success(f"📡 Data fetched securely from {PRIVATE_CONFIG['host']}")
    elif data_source=="Local CSV" and LOCAL_PATH:
        df = load_csv(LOCAL_PATH)
        st.success(f"📁 Loaded local data from: {LOCAL_PATH}")
    elif data_source=="Uploaded CSV":
        df = pd.read_csv(uploaded_file)
        st.success(f"📁 Loaded uploaded CSV: {uploaded_file.name}")
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

# --- Validate Columns ---
missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
for col in missing_cols: df[col] = 0

# Parse timestamps
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)

# Sidebar Filters
st.sidebar.header("Filters")
protocol_options = df['protocol'].unique().tolist() if 'protocol' in df.columns else []
default_protocols = [p for p in ['TCP','UDP','ICMP','OTHER'] if p in protocol_options]
protocol_filter = st.sidebar.multiselect("Protocol", options=protocol_options, default=default_protocols)
vt_risk_filter = st.sidebar.selectbox("VT Malicious Count > 0?", ['All','Yes','No'])
alert_filter = st.sidebar.selectbox("Only Alerts?", ['All','Yes','No'])
risk_filter = st.sidebar.slider("Minimum Combined Risk Score", 0, 100, 0)

# Apply Filters
if 'protocol' in df.columns and protocol_filter: df = df[df['protocol'].isin(protocol_filter)]
if 'vt_malicious_count' in df.columns:
    if vt_risk_filter=='Yes': df = df[df['vt_malicious_count']>0]
    elif vt_risk_filter=='No': df = df[df['vt_malicious_count']==0]
if 'final_alert' in df.columns:
    if alert_filter=='Yes': df = df[df['final_alert']==1]
    elif alert_filter=='No': df = df[df['final_alert']==0]
if 'combined_risk_score' in df.columns:
    df = df[df['combined_risk_score']>=risk_filter]

# --- AI Anomaly Detection (optional LSTM) ---
if not df.empty:
    features = ['length','ttl','flow_count','entropy_src','entropy_dst']
    df[features] = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    if X_scaled.size == 0 or np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
        st.warning("⚠️ AI anomaly detection skipped: Invalid feature data")
        df['ai_anomaly_score'] = 0
        df['ai_alert'] = 0
    else:
        # Isolation Forest
        iso = IsolationForest(contamination=0.05, random_state=42)
        df['ai_iso_anom'] = np.where(iso.fit_predict(X_scaled)==-1,1,0)

        if not fast_mode:
            # Optional LSTM
            seq = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            try:
                auto = Sequential([
                    LSTM(32, input_shape=(1,X_scaled.shape[1]), return_sequences=True),
                    Dropout(0.2),
                    LSTM(16, return_sequences=False),
                    Dense(X_scaled.shape[1], activation='linear')
                ])
                auto.compile(optimizer='adam', loss='mse')
                auto.fit(seq, X_scaled, epochs=1, batch_size=64, verbose=0)

                recon = auto.predict(seq)
                mse = np.mean(np.power(X_scaled - recon,2), axis=1)
                threshold = mse.mean() + 2*mse.std()
                df['ai_anomaly_score'] = np.round((mse/threshold)*100,2)
                df['ai_alert'] = (df['ai_anomaly_score']>80).astype(int)
            except Exception as e:
                st.warning(f"⚠️ LSTM anomaly detection failed: {e}")
                df['ai_anomaly_score'] = 0
                df['ai_alert'] = 0
        else:
            # Fast Mode: only Isolation Forest
            df['ai_anomaly_score'] = df['ai_iso_anom']*100
            df['ai_alert'] = df['ai_iso_anom']

        # Z-score anomaly
        if df['flow_count'].std() > 0:
            df['z_score'] = np.abs((df['flow_count'] - df['flow_count'].mean()) / df['flow_count'].std())
            df['z_alert'] = (df['z_score']>3).astype(int)
        else:
            df['z_score'] = 0
            df['z_alert'] = 0

        # Combined risk score
        df['combined_risk_score'] = np.round((df['risk_score'] + df['ai_anomaly_score'] + df['z_score']*10)/3,2)
        df['ai_suggestion'] = df['combined_risk_score'].apply(
            lambda x: "Monitor" if x<60 else ("Investigate" if x<85 else "Immediate Action")
        )

# --- IP Geolocation (cached, skipped in Fast Mode) ---
if not df.empty and not fast_mode:
    for ip_col in ['src','dst']:
        df[f"{ip_col}_city"] = None
        df[f"{ip_col}_country"] = None
        unique_ips = df[ip_col].unique()
        for ip in unique_ips[:50]:
            city, country = geolocate_ip(ip, geo_cache)
            df.loc[df[ip_col]==ip,f"{ip_col}_city"] = city
            df.loc[df[ip_col]==ip,f"{ip_col}_country"] = country
    with open(GEO_CACHE_FILE,"w") as f:
        json.dump(geo_cache, f)

# --- Dashboard Metrics & Visualizations ---
if df.empty:
    st.info("📭 No data available with current filters")
else:
    # Overview Metrics
    st.markdown("## 📊 Network Overview Metrics")
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Total Packets", len(df))
    col2.metric("Detected Alerts", int(df['final_alert'].sum()) if 'final_alert' in df.columns else 0)
    col3.metric("High VT Risk IPs", int((df['vt_malicious_count']>0).sum()) if 'vt_malicious_count' in df.columns else 0)
    col4.metric("Unique Source IPs", df['src'].nunique() if 'src' in df.columns else 0)

    # Top Talkers
    st.markdown("## 🔝 Top Talkers & Destinations")
    col1,col2 = st.columns(2)
    col1.bar_chart(df['src'].value_counts().head(10))
    col2.bar_chart(df['dst'].value_counts().head(10))

    # Cross-Network Traffic
    if 'cross_network' in df.columns:
        cross_pct = df['cross_network'].sum()/len(df)*100
        st.metric("Cross-Network Traffic %", f"{cross_pct:.2f}%")

    # Traffic Over Time
    if 'timestamp' in df.columns:
        traffic_over_time = df.groupby(pd.Grouper(key='timestamp', freq='1T')).size()
        st.markdown("## ⏱ Traffic Volume Over Time")
        st.line_chart(traffic_over_time)

    # AI Alerts Summary
    if 'ai_alert' in df.columns:
        st.markdown("## 🤖 AI Anomaly Alerts Summary")
        st.bar_chart(df[['ai_alert','ai_anomaly_score','z_alert']].sum())

    # High Risk Packets
    if 'combined_risk_score' in df.columns:
        high_risk = df[df['combined_risk_score']>80].sort_values('combined_risk_score',ascending=False)
        if not high_risk.empty:
            st.markdown("## 🔥 High Risk Packets (Score > 80)")
            st.dataframe(high_risk[['timestamp','src','dst','protocol','combined_risk_score','ai_suggestion']].head(25))

    # Protocol Distribution
    st.markdown("## 📡 Protocol Distribution")
    fig = px.histogram(df,x='protocol',
                       color='final_alert' if 'final_alert' in df.columns else None,
                       barmode='group' if 'final_alert' in df.columns else None,
                       title='Protocol vs Alerts' if 'final_alert' in df.columns else 'Protocol Distribution')
    st.plotly_chart(fig,use_container_width=True)

    # Packet Size vs TTL Heatmap
    st.markdown("## 🌡 Packet Size vs TTL Heatmap")
    fig2 = px.density_heatmap(df,x='ttl',y='length',
                              z='combined_risk_score' if 'combined_risk_score' in df.columns else None,
                              nbinsx=50,nbinsy=50,color_continuous_scale='Inferno')
    st.plotly_chart(fig2,use_container_width=True)

    # VirusTotal Summary
    if 'vt_malicious_count' in df.columns:
        st.markdown("## 🦠 VirusTotal Risk Summary")
        st.bar_chart(df['vt_malicious_count'].value_counts())

    # Recent Alerts
    st.markdown("## ⚠ Recent Alerts")
    if 'ai_alert' in df.columns and 'timestamp' in df.columns:
        recent_alerts = df[df['ai_alert']==1].sort_values('timestamp').tail(25)
        st.dataframe(recent_alerts)

    # Interactive Network Graph (top 25 for speed)
    st.markdown("## 🌐 Interactive Network Graph")
    top_ips = df[['src','dst','combined_risk_score']].copy()
    top_ips = top_ips.sort_values('combined_risk_score',ascending=False).head(25)
    if not top_ips.empty:
        G = nx.from_pandas_edgelist(top_ips,'src','dst',['combined_risk_score'])
        net = Network(height="600px", width="100%", notebook=False, bgcolor="#222222", font_color="white")
        net.from_nx(G)
        for e in net.edges:
            score = e.get('value',0)
            e['color'] = 'red' if score>80 else ('orange' if score>60 else 'green')
        tmp_file = "network_temp.html"
        net.write_html(tmp_file)
        with open(tmp_file, "r", encoding='utf-8') as f:
            components.html(f.read(), height=600)
    else:
        st.info("⚠️ Not enough data to generate network graph")

    # Download Filtered Data
    st.download_button(
        label="📥 Download Filtered Data",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name=f'filtered_results_{int(time.time())}.csv',
        mime='text/csv'
    )

# --- 💬 Gemini AI Chat Assistant ---
st.markdown("## 🧠 Gemini AI Chat Assistant for Network Analysis")
st.caption("Ask about traffic anomalies, suspicious IPs, or risk insights. Powered by Google Gemini AI.")

# --- Sidebar API Config ---
GEMINI_API_KEY = st.sidebar.text_input("AIzaSyDWhFYdrPM3rC8r2U2902dFTkQGu9WCOTE", type="password")
GEMINI_MODEL = st.sidebar.selectbox("Model", ["gemini-1.5-pro", "gemini-1.5-flash"], index=0)

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.sidebar.error(f"API Key error: {e}")

# --- Initialize chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous conversation
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# --- User input ---
user_input = st.chat_input("Ask about your network...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        if not GEMINI_API_KEY:
            st.error("⚠️ Please enter your Gemini API key in the sidebar first.")
        elif df.empty:
            st.warning("📊 No network data available to analyze.")
        else:
            try:
                # --- Context building from dataframe ---
                top_risks = df[df['combined_risk_score'] > 80][['src', 'dst']].head(5).to_dict('records')
                summary_context = f"""
                You are an expert cybersecurity network analyst.
                Current network context:
                - Total Packets: {len(df)}
                - Alerts: {df['final_alert'].sum() if 'final_alert' in df else 0}
                - Unique Source IPs: {df['src'].nunique() if 'src' in df else 0}
                - Average Risk Score: {df['combined_risk_score'].mean():.2f}
                - Top Risky IP Pairs: {top_risks}
                """

                # --- Combine context + chat memory ---
                chat_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-5:]])
                full_prompt = textwrap.dedent(f"""
                {summary_context}

                Previous conversation:
                {chat_context}

                User Question: {user_input}

                Respond like a professional SOC analyst.
                Explain findings clearly, highlight anomalies, and give actionable mitigations.
                Use structured markdown if useful.
                """)

                # --- Stream AI response ---
                model = genai.GenerativeModel(GEMINI_MODEL)
                with st.spinner("Analyzing with Gemini..."):
                    response = model.generate_content(full_prompt, stream=True)
                    ai_reply = ""
                    for chunk in response:
                        if hasattr(chunk, "text"):
                            ai_reply += chunk.text
                            st.markdown(chunk.text)

                # Store in session
                st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})

            except Exception as e:
                st.error(f"Gemini AI error: {e}")