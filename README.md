# Package Delivery Delay Estimator 
Package Delivery Delay Estimator predicts the number of days a package will be delayed based on order logistics (courier type, distance, shipping delay, cities, etc.), and visualize delay risk per city.

## Features :hammer_and_wrench:
- Orders can be uoloaded via CSV file
- Visual Anaylytics to delay distribution and predicted delay days
- Summary tables by courier and city
- Interactive map of delivery delays by destination city
- Ready to download CSV file containing prediction of delay in days and delay level

## Demo 📸
![Demo](./Demo/Demo.gif)

## Tech Stack
- Python
- Streamlit - Web application framework for Python
- HTML/CSS - Custom styling and UI components
- Model - RandomForestRegressor
- Metrics - MAE,R²



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
