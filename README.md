# flood_prediction_project
เเจ้งเตือนนำ้ท่วมในประเทศไทยเชิงพื้นที่ด้วยการใช้ข้อมูลทางดาวเทียม
แน่นอนครับ! นี่คือฉบับร่างของไฟล์ README.md ที่คุณสามารถใช้สำหรับโปรเจกต์ flood_prediction_project ของคุณบน GitHub ได้เลยครับ ผมได้ใส่ข้อมูลที่คุณต้องการและโครงสร้างพื้นฐานที่ทำให้ README น่าอ่านและเป็นประโยชน์ครับ

Flood Prediction Project
💡 Project Overview
This project focuses on developing a robust machine learning model to predict flood events and water levels, aiming to provide early warnings and support disaster preparedness. By integrating diverse data sources, including historical weather, satellite imagery, and geographical features, the model learns complex patterns to forecast flood risks.

🚀 Key Features
Data Ingestion: Collects and integrates historical weather data from various sources, including OpenWeatherMap (OWM) and potentially satellite data via Google Earth Engine (GEE), alongside simulated historical data.
Feature Engineering: Extracts crucial features such as rainfall accumulation, temperature, humidity, and lunar phase (using a simplified approximation) to enhance model performance.
Data Preprocessing: Handles missing values, scales numerical features, and prepares data for machine learning model training.
Model Training: Utilizes advanced machine learning techniques, specifically Long Short-Term Memory (LSTM) neural networks, for both classification (predicting flood occurrence) and regression (predicting water levels).
Prediction Service: Provides a framework for real-time flood prediction based on the trained models.
🛠️ Technologies Used
Python: Core programming language.
Pandas: For data manipulation and analysis.
NumPy: For numerical operations.
TensorFlow/Keras: For building and training neural networks (LSTM).
Scikit-learn: For data preprocessing (e.g., MinMaxScaler) and model evaluation.
pytz: For robust timezone handling.
joblib: For saving and loading models and scalers.
OpenWeatherMap API: For weather data ingestion.
Google Earth Engine (GEE): (Conceptual, if fully integrated for satellite data).
🚀 Getting Started
Follow these steps to set up and run the project locally.

📋 Prerequisites
Python 3.8+
pip (Python package installer)
Git
📦 Installation
Clone the repository:
Bash

git clone https://github.com/Teeraphat2543Inta/flood_prediction_project.git
cd flood_prediction_project
Create a virtual environment (recommended):
Bash

python -m venv venv
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
Install the required packages:
Bash

pip install -r requirements.txt
⚙️ Configuration
Create a config.py file: Make sure your config.py file (located in the config/ directory) is properly set up with your API keys (e.g., OpenWeatherMap API key) and other necessary configurations like timezones and target columns.
Python

# config/config.py (Example content)
import os

# API Keys
OWM_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY" # Replace with your actual key

# Timezone for local operations
TIMEZONE = 'Asia/Bangkok'

# Data paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw')
HISTORICAL_WEATHER_DATA_PATH = os.path.join(DATA_RAW_PATH, 'historical_weather_data.csv')

# Model paths
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True) # Ensure models directory exists

# Define paths for classification model
CLASSIFICATION_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'classification_model.h5')
CLASSIFICATION_SCALER_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'classification_scaler.joblib')
CLASSIFICATION_FEATURES_ORDER_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'classification_features_order.joblib')

# Define paths for regression model
REGRESSION_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'regression_model.h5')
REGRESSION_SCALER_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'regression_scaler.joblib')
REGRESSION_FEATURES_ORDER_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'regression_features_order.joblib')

# Model Training Configuration
# You can choose to train 'classification', 'regression', or both ['classification', 'regression']
MODEL_TYPES_TO_TRAIN = ['classification'] # Example: train only classification initially

# Target column names
TARGET_CLASSIFICATION = 'is_flood_next_1h'
TARGET_REGRESSION = 'water_level_next_1h'

# Features (adjust based on your dataset)
# Ensure these match the columns generated in your data preprocessing
FEATURES = [
    'temperature', 'humidity', 'wind_speed', 'precipitation_1h',
    'pressure', 'cloud_cover', 'dew_point', 'visibility',
    'temp_min', 'temp_max',
    'rainfall_3h_accum', 'rainfall_6h_accum', 'rainfall_12h_accum', 'rainfall_24h_accum',
    'temp_avg_6h', 'humidity_avg_6h',
    'is_day', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
    'month_sin', 'month_cos', 'lunar_phase'
    # Add other GEE-derived features if applicable
]
▶️ Running the Project
Collect Historical Data & Train Model:
This script will collect (or use simulated) historical data, preprocess it, and then train and save the machine learning model(s) as configured in config.py.

Bash

python main.py
This process will save the trained model (.h5), scaler (.joblib), and feature order (.joblib) to the models/ directory.

Run the Prediction Service:
Once the model is trained, you can start the prediction service to make real-time inferences.
(Ensure config.py's CURRENT_PREDICTION_MODEL_TYPE is set to the model you want to load, e.g., 'classification' or 'regression')

Bash

python src/prediction_service.py
📈 Model Evaluation
The main.py script will output evaluation metrics after training.

For Classification Models: Metrics like Accuracy, Precision, Recall, F1-Score, and ROC-AUC are typically reported.
For Regression Models: Metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) are relevant.
🧑‍💻 Contributing
Contributions are welcome! Please feel free to fork the repository, make improvements, and submit pull requests.

📄 License
This project is open-source and available under the MIT License.

📞 Contact
For any questions or inquiries, please reach out to:

Teeraphat James Inta PhD Candidate at MIT Email: Teerapat2543jame@gmail.com

LinkedIn: www.linkedin.com/in/teeraphat-inta-bb57b1277



