# streamlit_live_capture.py
import streamlit as st
import pandas as pd
import os, time
from datetime import datetime
from scapy.all import sniff, IP, conf
from collections import Counter

conf.L3socket = conf.L3RawSocket
os.makedirs('data', exist_ok=True)
CSV_PATH = 'data/realtime_network.csv'
MAX_ROWS = 5000

packet_counter = Counter()

def entropy_of_string(s: str) -> float:
    if not s: return 0.0
    from collections import Counter
    import math
    counts = Counter(s)
    probs = [c/len(s) for c in Counter(s).values()]
    return -sum(p*math.log2(p) for p in probs)

def process_packet(pkt):
    if IP not in pkt: return None
    ip = pkt[IP]
    proto = {6:'TCP',17:'UDP',1:'ICMP'}.get(ip.proto,'OTHER')
    key = (ip.src, ip.dst, proto)
    packet_counter[key] += 1
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'src': ip.src,
        'dst': ip.dst,
        'protocol': proto,
        'length': len(pkt),
        'ttl': int(ip.ttl),
        'flow_count': packet_counter[key],
        'entropy_src': entropy_of_string(ip.src),
        'entropy_dst': entropy_of_string(ip.dst)
    }

def capture_batch(iface, count):
    rows = []
    sniff(iface=iface, prn=lambda pkt: rows.append(process_packet(pkt)) if process_packet(pkt) else None, count=count)
    return rows

def append_to_csv(rows):
    df_new = pd.DataFrame(rows)
    if os.path.exists(CSV_PATH):
        df_existing = pd.read_csv(CSV_PATH)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        if len(df_combined) > MAX_ROWS:
            df_combined = df_combined[-MAX_ROWS:]
        df_combined.to_csv(CSV_PATH, index=False)
    else:
        df_new.to_csv(CSV_PATH, index=False)

# --- Streamlit Dashboard ---
st.title("🧠 Live Network Capture Dashboard")
iface = st.sidebar.text_input("Interface", "Wi-Fi")
count = st.sidebar.number_input("Packets per batch", 100, 2000, 500)
start_capture = st.sidebar.button("Start Capture")

placeholder = st.empty()

if start_capture:
    try:
        while True:
            rows = capture_batch(iface, count)
            if rows:
                append_to_csv(rows)
                df = pd.read_csv(CSV_PATH)
                with placeholder.container():
                    st.metric("Total Packets", len(df))
                    st.metric("Unique Source IPs", df['src'].nunique())
                    st.dataframe(df.tail(20))
            time.sleep(2)
    except KeyboardInterrupt:
        st.warning("Capture stopped")
