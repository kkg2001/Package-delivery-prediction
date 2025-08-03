import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

st.set_page_config(page_title="Delivery Delay Estimator", layout="wide")
st.title("📦 Delivery Delay Estimator")

st.sidebar.header("Upload Order CSV")
file = st.sidebar.file_uploader("Upload a CSV with order features", type=["csv"])

@st.cache_resource
def load_model():
    return joblib.load("model/model.pkl")

model = load_model()

@st.cache_data
def get_city_coords(cities):
    geolocator = Nominatim(user_agent="delay_map")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    coords = []
    for city in cities:
        location = geocode(city)
        if location:
            coords.append((city, location.latitude, location.longitude))
        else:
            coords.append((city, None, None))
    return pd.DataFrame(coords, columns=['Destination City', 'lat', 'lon'])

if file is not None:
    df = pd.read_csv(file)

    required_cols = ['courier_type', 'distance_km', 'shipping_delay_days', 'source_city', 'destination_city']
    if set(required_cols).issubset(df.columns):
        df['predicted_delay_days'] = model.predict(df[required_cols])

        def label_delay(x):
            if x == 0:
                return '🟢 On Time'
            elif x <= 2:
                return '🟡 Slight Delay'
            elif x <= 5:
                return '🟠 Moderate Delay'
            else:
                return '🔴 Severe Delay'

        df['delay_level'] = df['predicted_delay_days'].round().apply(label_delay)

        st.success("✅ Delay prediction completed!")
        st.subheader("📄 Predicted Delivery Delays")
        st.dataframe(df[['courier_type', 'distance_km', 'shipping_delay_days', 'source_city', 'destination_city', 'predicted_delay_days', 'delay_level']].head(15))

        selected_levels = st.multiselect(
            "Filter by Delay Level:",
            options=df['delay_level'].unique(),
            default=df['delay_level'].unique()
        )
        filtered_df = df[df['delay_level'].isin(selected_levels)]

        st.subheader("📊 Predicted Delay Distribution")
        fig, ax = plt.subplots()
        sns.histplot(filtered_df['predicted_delay_days'], bins=20, kde=True, ax=ax, color='orange')
        ax.set_xlabel("Predicted Delay (Days)")
        st.pyplot(fig)

        st.subheader("📈 Delay Category Counts")
        st.bar_chart(filtered_df['delay_level'].value_counts())

        if st.checkbox("📋 Show Delay Summary Tables", value=True):
            st.subheader("📍 Average Delay by Courier Type")
            courier_avg = filtered_df.groupby('courier_type')['predicted_delay_days'].mean().reset_index()
            courier_avg.columns = ['Courier Type', 'Avg Delay (days)']
            st.dataframe(courier_avg)

            st.subheader("🌍 Average Delay by Destination City")
            city_avg = filtered_df.groupby('destination_city')['predicted_delay_days'].mean().reset_index()
            city_avg.columns = ['Destination City', 'Avg Delay (days)']
            st.dataframe(city_avg)

        if st.checkbox("🗺️ Show Delay Map by Destination City"):
            st.subheader("📍 Interactive Delay Map")
            city_avg = filtered_df.groupby('destination_city')['predicted_delay_days'].mean().reset_index()
            city_avg.columns = ['Destination City', 'Avg Delay (days)']

            city_coords = get_city_coords(city_avg['Destination City'].unique())
            city_avg = pd.merge(city_avg, city_coords, on='Destination City', how='left')
            map_data = city_avg.dropna(subset=['lat', 'lon'])

            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=pdk.ViewState(
                    latitude=map_data['lat'].mean(),
                    longitude=map_data['lon'].mean(),
                    zoom=4,
                    pitch=30,
                ),
                layers=[
                    pdk.Layer(
                        'ScatterplotLayer',
                        data=map_data,
                        get_position='[lon, lat]',
                        get_color='[255, 140, 0, 140]',
                        get_radius='["Avg Delay (days)"] * 5000',
                        pickable=True
                    )
                ],
                tooltip={"text": "{Destination City}: {Avg Delay (days)} days"}
            ))

        csv_out = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download Results as CSV",
            data=csv_out,
            file_name="predicted_delivery_delays_filtered.csv",
            mime="text/csv"
        )

    else:
        st.error("CSV must include required columns: courier_type, distance_km, shipping_delay_days, source_city, destination_city")
else:
    st.info("Upload CSV file to get delivery delay predictions.")
