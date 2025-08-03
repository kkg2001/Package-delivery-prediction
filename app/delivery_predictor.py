import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pydeck as pdk
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

st.set_page_config(page_title="Package Delivery Delay Predictor", layout="wide")

@st.cache_resource
def load_model():
    try:
        return joblib.load("model\model.pkl")
    except Exception as e:
        st.error("❌ Failed to load model.pkl. Check path or retrain it.")
        st.stop()

model = load_model()

def load_metrics():
    if os.path.exists("model\metrics.txt"):
        with open("model\metrics.txt") as f:
            return f.read()
    return "No training metrics found."

@st.cache_data
def get_city_coords(cities):
    geolocator = Nominatim(user_agent="delivery_app")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    coords = []
    for city in cities:
        location = geocode(city)
        if location:
            coords.append((city, location.latitude, location.longitude))
        else:
            coords.append((city, None, None))
    return pd.DataFrame(coords, columns=["destination_city", "lat", "lon"])

st.title("📦 Delivery Delay Predictor")
st.markdown("Upload delivery data to predict expected delay in **days** per shipment.")

st.sidebar.header("⚙️ Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_cols = ['courier_type', 'distance_km', 'shipping_delay_days', 'source_city', 'destination_city']

    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns. Ensure your CSV includes: {required_cols}")
        st.stop()

    preds = model.predict(df[required_cols])
    df['predicted_delay_days'] = preds.round(2)

    def classify_level(d):
        if d == 0:
            return "🟢 On Time"
        elif d <= 2:
            return "🟡 Slight Delay"
        elif d <= 5:
            return "🟠 Moderate Delay"
        else:
            return "🔴 Severe Delay"

    df['delay_level'] = df['predicted_delay_days'].round().apply(classify_level)

    st.success("✅ Delay prediction completed!")
    st.subheader("📄 Predicted Delivery Delays")
    st.dataframe(df[['courier_type', 'distance_km', 'shipping_delay_days', 'source_city', 'destination_city', 'predicted_delay_days', 'delay_level']].head(15))

    selected_levels = st.multiselect(

        "Filter by Delay Level:",
        options=df['delay_level'].unique(),
        default=df['delay_level'].unique()
    )
    filtered_df = df[df['delay_level'].isin(selected_levels)]

    st.download_button("⬇️ Download Results CSV", data=df.to_csv(index=False), file_name="predicted_delays.csv")

    st.subheader("📊 Delay Distribution")
    st.bar_chart(df['delay_level'].value_counts())

    st.subheader("📈 Avg Delay by Courier")
    st.dataframe(df.groupby("courier_type")["predicted_delay_days"].mean().reset_index())

    st.subheader("📊 Predicted Delay Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['predicted_delay_days'], bins=20, kde=True, ax=ax, color='orange')
    ax.set_xlabel("Predicted Delay (Days)")
    st.pyplot(fig)

    st.subheader("🌍 Avg Delay by Destination City")
    avg_city = df.groupby("destination_city")["predicted_delay_days"].mean().reset_index()
    coords = get_city_coords(avg_city["destination_city"].unique())
    avg_city = avg_city.merge(coords, on="destination_city", how="left").dropna()

    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=37.773972, longitude=-122.431297, zoom=3, pitch=0),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=avg_city,
                get_position='[lon, lat]',
                get_radius='predicted_delay_days * 4000',
                get_fill_color='[255, 100 - predicted_delay_days * 10, 100]',
                pickable=True,
                radius_min_pixels=5,
                radius_max_pixels=50,
            )
        ],
        tooltip={"text": "{destination_city}: {predicted_delay_days} days"},
    ))

    st.sidebar.subheader("📉 Model Performance")
    st.sidebar.code(load_metrics())

else:
    st.info("Upload a CSV file to begin.")
