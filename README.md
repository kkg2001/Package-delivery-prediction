# Package Delivery Delay Estimator 
Package Delivery Delay Estimator predicts the number of days a package will be delayed based on order logistics (courier type, distance, shipping delay, cities, etc.), and visualize delay risk per city.

## Features :hammer_and_wrench:
- Upload orders (new or historical)
- Predict number of days delayed
- Categorize delays (🟢 On Time → 🔴 Severe Delay)
- Charts: delay distribution, breakdowns
- Summary tables by courier and city
- Interactive map of delivery delays
- Download enriched prediction data

## Demo 📸
![Demo](./Demo/Demo.gif)

## Tech Stack
- Python
- Streamlit - Web application framework for Python
- HTML/CSS - Custom styling and UI components
- JavaScript (via Streamlit) - Interactive elements
- JSON - Data storage format

## How to Run
# Step 1: Install the requirements
pip install -r requirements.txt

# Step 2: Train the model
python preprocessing.py

# Step 3:  Launch the Application
streamlit run app/delivery_predictor.py

## Future Ideas
- Use weather/holiday info to enhance delay prediction
- Export maps and plots as images
- Add login/user profiles to track predictions
