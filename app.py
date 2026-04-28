import streamlit as st
import pandas as pd
import pickle

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Factory Optimization", page_icon="🏭", layout="wide")

# -----------------------------
# STYLE
# -----------------------------
st.markdown("""
<style>
.stApp {background-color: #0E1117; color: white;}
section[data-testid="stSidebar"] {background-color: #1E1E1E;}
.stButton>button {background-color: #6C63FF; color: white; border-radius: 10px;}
div[data-testid="metric-container"] {background-color: #1E1E1E; padding: 10px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# -----------------------------
# CONSTANTS
# -----------------------------
factories = [
    "Lot's O' Nuts",
    "Wicked Choccy's",
    "Sugar Shack",
    "Secret Factory",
    "The Other Factory"
]

products = [col.replace("Product Name_", "") for col in columns if "Product Name_" in col]
regions = ["Gulf", "Interior", "Pacific"]
ship_modes = ["Standard Class", "Second Class", "Same Day"]

# -----------------------------
# SIMULATION FUNCTION
# -----------------------------
def simulate(product, region, ship_mode):
    results = []

    for factory in factories:
        input_data = pd.DataFrame(columns=columns)
        input_data.loc[0] = 0

        if f"Product Name_{product}" in columns:
            input_data[f"Product Name_{product}"] = 1

        if f"Ship Mode_{ship_mode}" in columns:
            input_data[f"Ship Mode_{ship_mode}"] = 1

        if f"Factory_{factory}" in columns:
            input_data[f"Factory_{factory}"] = 1

        if f"Region_{region}" in columns:
            input_data[f"Region_{region}"] = 1

        pred = model.predict(input_data)[0]

        results.append({
            "Factory": factory,
            "Lead Time": round(pred, 2)
        })

    df = pd.DataFrame(results)

    # Fake profit estimation (simple logic)
    df["Estimated Profit"] = 1000 / df["Lead Time"]

    worst = df["Lead Time"].max()
    df["Improvement"] = worst - df["Lead Time"]

    return df.sort_values("Lead Time")

# -----------------------------
# CLUSTERING
# -----------------------------
def cluster_routes(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[["Lead Time"]])

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(scaled)

    return df

# -----------------------------
# HEADER
# -----------------------------
st.title("🏭 Factory Optimization Dashboard")
st.markdown("### AI-powered shipping & profit optimization")

st.markdown("---")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Controls")

product = st.sidebar.selectbox("📦 Product", products)
region = st.sidebar.selectbox("🌍 Region", regions)
ship_mode = st.sidebar.selectbox("🚚 Ship Mode", ship_modes)

# 🎚 NEW SLIDER
priority = st.sidebar.slider("🎯 Optimization Priority (Speed ↔ Profit)", 0, 100, 50)

run = st.sidebar.button("🚀 Run Simulation")

# -----------------------------
# MAIN
# -----------------------------
if run:

    result = simulate(product, region, ship_mode)
    result = cluster_routes(result)

    # -----------------------------
    # APPLY PRIORITY LOGIC
    # -----------------------------
    speed_weight = (100 - priority) / 100
    profit_weight = priority / 100

    result["Score"] = (
        speed_weight * result["Lead Time"] * -1 +
        profit_weight * result["Estimated Profit"]
    )

    result = result.sort_values("Score", ascending=False)

    best = result.iloc[0]
    worst = result.iloc[-1]

    # -----------------------------
    # KPI
    # -----------------------------
    st.subheader("📊 Key Performance Indicators")

    lead_reduction = round(((worst["Lead Time"] - best["Lead Time"]) / worst["Lead Time"]) * 100, 2)
    avg_profit = round(result["Estimated Profit"].mean(), 2)

    col1, col2, col3 = st.columns(3)
    col1.metric("📉 Lead Time Reduction (%)", lead_reduction)
    col2.metric("💰 Avg Profit Score", avg_profit)
    col3.metric("🏆 Best Factory", best["Factory"])

    st.markdown("---")

    # -----------------------------
    # TABLE
    # -----------------------------
    st.subheader("📋 Factory Performance")
    st.dataframe(result, use_container_width=True)

    # -----------------------------
    # CHARTS
    # -----------------------------
    st.subheader("📊 Lead Time Comparison")
    st.bar_chart(result.set_index("Factory")["Lead Time"])

    st.subheader("📈 Profit Comparison")
    st.line_chart(result.set_index("Factory")["Estimated Profit"])

    # -----------------------------
    # TOP RECOMMENDATIONS
    # -----------------------------
    st.subheader("🏆 Top Recommendations")
    st.dataframe(result.head(3))

    st.success(f"Best Factory: {best['Factory']}")

    # -----------------------------
    # PROFIT IMPACT
    # -----------------------------
    st.subheader("💰 Profit Impact Analysis")

    st.write(f"Best Factory Profit Score: {round(best['Estimated Profit'],2)}")
    st.write(f"Worst Factory Profit Score: {round(worst['Estimated Profit'],2)}")

    # -----------------------------
    # CLUSTERING
    # -----------------------------
    st.subheader("🔍 Route Clustering")

    slow_routes = result[result["Cluster"] == result["Cluster"].max()]

    if not slow_routes.empty:
        st.warning("Slow routes detected")
        st.dataframe(slow_routes)

    # -----------------------------
    # WHAT-IF
    # -----------------------------
    st.subheader("🔄 What-If Analysis")
    st.write(f"Improvement: {round(worst['Lead Time'] - best['Lead Time'],2)} days")

    # -----------------------------
    # RISK PANEL
    # -----------------------------
    st.subheader("⚠️ Risk Panel")

    if best["Lead Time"] > 7:
        st.warning("High lead time risk")
    else:
        st.success("Optimized configuration")