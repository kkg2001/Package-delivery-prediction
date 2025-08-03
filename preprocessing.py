import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# --- Load Data ---
df = pd.read_csv("data/orders.csv")

# --- Ensure required columns exist or simulate them ---
if 'shipping_delay_days' not in df.columns:
    df['shipping_delay_days'] = (df['distance_km'] / 500).round().astype(int)

df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
df['ship_date'] = pd.to_datetime(df['ship_date'], errors='coerce')
df['expected_delivery_date'] = df['ship_date'] + pd.to_timedelta(df['shipping_delay_days'], unit='D')

if 'actual_delivery_date' not in df.columns:
    df['actual_delay_days'] = df['shipping_delay_days'] + (df['distance_km'] / 500).round().astype(int)
    df['actual_delivery_date'] = df['expected_delivery_date'] + pd.to_timedelta(df['actual_delay_days'], unit='D')
else:
    df['actual_delivery_date'] = pd.to_datetime(df['actual_delivery_date'], errors='coerce')
    df['actual_delay_days'] = (df['actual_delivery_date'] - df['expected_delivery_date']).dt.days

df = df[df['actual_delay_days'].notnull() & (df['actual_delay_days'] >= 0)]

sns.histplot(df['actual_delay_days'], bins=20, kde=True)
plt.title("Actual Delivery Delay (Days)")
plt.tight_layout()
os.makedirs("data", exist_ok=True)
plt.savefig("data/plot_actual_delay_distribution.png")
plt.clf()

features = ['courier_type', 'distance_km', 'shipping_delay_days', 'source_city', 'destination_city']
target = 'actual_delay_days'
X = df[features]
y = df[target]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ['courier_type', 'source_city', 'destination_city']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Regression Model Performance")
print(f"Mean Absolute Error: {mae:.2f} days")
print(f"R² Score: {r2:.2f}")

os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/model.pkl", compress=3, protocol=4)
with open("model/metrics.txt", "w") as f:
    f.write(f"MAE: {mae:.2f}\nR²: {r2:.2f}")
print("\n✅ Regression model saved to model/model.pkl")
