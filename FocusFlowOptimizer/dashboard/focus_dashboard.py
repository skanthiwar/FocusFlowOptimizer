"""
Focus Flow Optimizer — Live Dashboard (Streamlit)
------------------------------------------------
Visualizes focus predictions and activity in real time from session_log.csv
Run with:
    streamlit run dashboard/focus_dashboard.py
"""

import os
import time
import pandas as pd
import streamlit as st
import plotly.express as px

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION_LOG = os.path.join(PROJECT_ROOT, "data", "session_log.csv")

# --- Page Config ---
st.set_page_config(page_title="Focus Flow Dashboard", page_icon="🧠", layout="wide")

# --- Header ---
st.title("🧠 Focus Flow Optimizer — Live Dashboard")
st.caption("Tracking your real-time focus and activity trends")

# --- Helper Function ---
@st.cache_data(ttl=5)
def load_data():
    if not os.path.exists(SESSION_LOG):
        return pd.DataFrame(columns=["Timestamp", "Active_App", "Key_Rate", "Mouse_Rate", "Predicted_State", "Confidence"])
    df = pd.read_csv(SESSION_LOG)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.sort_values("Timestamp")
    return df.tail(500)  # show recent 500 rows

# --- Sidebar Controls ---
refresh_rate = st.sidebar.slider("🔄 Refresh every (seconds)", 3, 30, 5)
show_apps = st.sidebar.multiselect("🎯 Filter by App", [], help="Select apps to filter once loaded")

# --- Main Loop (Auto-Refresh) ---
placeholder = st.empty()

while True:
    df = load_data()
    if df.empty:
        st.warning("No session data yet. Run `flow_optimizer.py` first.")
        time.sleep(refresh_rate)
        st.experimental_rerun()

    # Auto-populate filter options
    if not show_apps:
        unique_apps = sorted(df["Active_App"].dropna().unique().tolist())
        show_apps = st.sidebar.multiselect("🎯 Filter by App", unique_apps, default=unique_apps)

    filtered = df[df["Active_App"].isin(show_apps)] if show_apps else df

    # Layout: Metrics + Charts
    with placeholder.container():
        col1, col2, col3 = st.columns(3)
        if not filtered.empty:
            latest = filtered.iloc[-1]
            col1.metric("🕒 Last Update", latest["Timestamp"].strftime("%H:%M:%S"))
            col2.metric("🧩 Last Prediction", latest["Predicted_State"])
            col3.metric("🎯 Confidence", f"{latest['Confidence']*100:.1f}%")
        else:
            st.info("Waiting for new focus data...")

        st.divider()
        st.subheader("⏱️ Focus Over Time")

        if not filtered.empty:
            fig1 = px.line(
                filtered,
                x="Timestamp",
                y="Confidence",
                color="Predicted_State",
                title="Focus Probability Over Time",
                markers=True,
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)

            st.divider()
            st.subheader("⌨️ Input Activity Trends")

            fig2 = px.line(
                filtered,
                x="Timestamp",
                y=["Key_Rate", "Mouse_Rate"],
                title="Keyboard & Mouse Activity",
                markers=True,
            )
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

            st.divider()
            st.subheader("🪟 Active Apps Distribution")

            app_counts = filtered["Active_App"].value_counts().reset_index()
            app_counts.columns = ["App", "Count"]
            fig3 = px.bar(app_counts, x="App", y="Count", title="Most Active Apps", text="Count")
            fig3.update_layout(height=300)
            st.plotly_chart(fig3, use_container_width=True)

        time.sleep(refresh_rate)
        st.experimental_rerun()

def load_data():
    if not os.path.exists(SESSION_LOG):
        return pd.DataFrame(columns=["Timestamp", "Active_App", "Key_Rate", "Mouse_Rate", "Predicted_State", "Confidence"])
    df = pd.read_csv(SESSION_LOG)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.sort_values("Timestamp")
    return df.tail(500)
