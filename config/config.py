# config/config.py

import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# โหลดตัวแปรสภาพแวดล้อมจากไฟล์ .env
load_dotenv() 

#  For example: path_to_json_key = ./key/benz-gee-50f789454788.json
GEE_JSON_KEY = os.getenv("path_to_json_key")

# --- API Keys ---
# ดึง API Key จากตัวแปรสภาพแวดล้อมเพื่อความปลอดภัย
# หากไม่มีค่าใน .env จะใช้ค่าเริ่มต้นที่เป็น placeholder
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "YOUR_OPENWEATHER_API_KEY_HERE")
# LOCAL_WATER_LEVEL_API_KEY = os.getenv("LOCAL_WATER_LEVEL_API_KEY", "YOUR_LOCAL_WATER_LEVEL_API_KEY") 
# ^^^ [แก้ไข] ถ้าคุณมี API Key สำหรับข้อมูลระดับน้ำในพื้นที่จริง ให้เปิดบรรทัดนี้และใส่ตัวแปรใน .env

# --- Database / Data Paths ---
# [แก้ไข] ปรับให้ใช้ BASE_DIR เพื่อให้ Path เป็น Relative กับ Root Project Folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # flood_prediction_model/

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

HISTORICAL_DATA_FILE = os.path.join(RAW_DATA_DIR, "historical_weather_data.csv") 
# ^^^ [แก้ไข] เปลี่ยนชื่อตัวแปรให้สอดคล้องกับโครงสร้าง project folder (จาก PATH_TO_HISTORICAL_DATA)
PROCESSED_TRAINING_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "processed_training_data.csv")
# ^^^ [แก้ไข] เปลี่ยนชื่อตัวแปรให้สอดคล้องกับโครงสร้าง project folder (จาก PROCESSED_DATA_PATH)

# --- Model Paths ---
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "best_flood_model.h5")
SCALER_SAVE_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_ORDER_SAVE_PATH = os.path.join(MODEL_DIR, "features_order.pkl")

# --- Geographic Information ---
TARGET_AREA_LAT = 13.7563  # Latitude for Bangkok (average)
TARGET_AREA_LON = 100.5018 # Longitude for Bangkok (average)
TIMEZONE = "Asia/Bangkok" 

# --- OpenWeatherMap API Endpoints ---
# [แก้ไข] เพิ่ม OWM_HISTORY_URL สำหรับการดึงข้อมูลย้อนหลัง (ถ้ามีสิทธิ์)
OWM_FORECAST_URL = "http://api.openweathermap.org/data/2.5/forecast" 
OWM_HISTORY_URL = "http://api.openweathermap.org/data/2.5/onecall/history" 
# ^^^ [เพิ่ม] จำเป็นสำหรับดึงข้อมูลย้อนหลัง ถ้าคุณมีแพลนที่เข้าถึงได้

# --- Google Earth Engine (GEE) Configuration ---
GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID", None) 
# ^^^ [แก้ไข] ควรดึงจาก .env เหมือน API Key อื่นๆ
GEE_SCALE_METERS = 30 

# --- Satellite Data Collections (GEE ImageCollection IDs) ---
# [แก้ไข] ชื่อ Collection และ Band ของ GPM IMERG ที่คุณระบุก่อนหน้านี้ถูกต้องแล้ว
# ตรวจสอบอีกครั้งใน GEE Data Catalog ว่าเป็นชื่อล่าสุดหรือไม่
GPM_IMERG_COLLECTION = "NASA/GPM_L3/IMERG_V06/HOURLY_V06" 

NASA_GPM_L3_IMERG_V07 = "NASA/GPM_L3/IMERG_V07"
# GPM_RAINFALL_BAND = "precipitationCal" 
GPM_RAINFALL_BAND = "precipitation" 


SENTINEL1_COLLECTION = "COPERNICUS/S1_GRD"
SENTINEL1_BAND = "VV" 
SENTINEL1_FILTER_MODE = "IW" 

SENTINEL2_COLLECTION = "COPERNICUS/S2_SR" 
SENTINEL2_BAND_GREEN = "B3" 
SENTINEL2_BAND_NIR = "B8" 
SENTINEL2_CLOUD_THRESHOLD = 20 

SMAP_COLLECTION = "NASA/SMAP/SPL3SMP_E/006" 
SMAP_SOIL_MOISTURE_BAND = "sm_rootzone_depth" 
# ^^^ [แก้ไข] ตรวจสอบว่า `sm_rootzone_depth` เป็น Band ที่คุณต้องการจริงๆ (มี sm_surface_moisture ด้วย)

# --- Flood Thresholds ---
FLOOD_WATER_LEVEL_THRESHOLD_M = 1.5 
# [เพิ่ม] เกณฑ์น้ำฝน และความชื้นในดิน เพื่อใช้ในการสร้าง `is_flood`
RAIN_THRESHOLD_FOR_FLOOD_MM_HR = 20.0 # ตัวอย่าง: ฝน 20 mm/hr ขึ้นไป ถือว่าตกหนักมาก
SOIL_MOISTURE_FLOOD_THRESHOLD = 0.45 # ตัวอย่าง: ค่าความชื้นในดินที่สูง (0-0.5, 0.45 คืออิ่มตัวมาก)

# --- Data Collection Parameters ---
# [เพิ่ม] กำหนดจำนวนวันย้อนหลังที่ต้องการดึงข้อมูล
HISTORICAL_DAYS_TO_FETCH = 365 * 2 # ตัวอย่าง: ดึงข้อมูลย้อนหลัง 2 ปี

# --- Model Parameters ---
SEQUENCE_LENGTH = 24 # [แก้ไข] ปกติจะใช้ข้อมูลย้อนหลัง 24 ชั่วโมง หรือ 48 ชั่วโมง เพื่อดูแนวโน้ม
# ^^^ หากคุณต้องการพยากรณ์ระดับน้ำ/น้ำท่วม การดูข้อมูลอย่างน้อย 24 ชั่วโมงย้อนหลังมีประโยชน์มาก
PREDICTION_HORIZON_HOURS = 1 # พยากรณ์ล่วงหน้า 1 ชั่วโมง

# [เพิ่ม] รายชื่อ Features ที่จะใช้ในการฝึกโมเดล (ก่อนที่จะสร้าง lagged features)
# ชื่อเหล่านี้จะต้องตรงกับชื่อคอลัมน์ใน historical_weather_data.csv
FEATURES = [
    "temperature", "humidity", "pressure", "wind_speed", "rain_1h", 
    "sea_level_m", "local_water_level_m", 
    "satellite_rainfall_mm_hr", 
    "satellite_water_index_s1_vv", # ถ้าใช้ Sentinel-1
    "satellite_water_index_s2_ndwi", # ถ้าใช้ Sentinel-2
    "satellite_soil_moisture"
]
TARGET = "is_flood" # ชื่อคอลัมน์ที่เป็น Target variable

# --- Training Parameters ---
EPOCHS = 100        
BATCH_SIZE = 32     
PATIENCE = 10       
LEARNING_RATE = 0.001 
TRAIN_TEST_SPLIT_RATIO = 0.8 # [เพิ่ม] อัตราส่วน Train/Test Split

# --- Model Architecture Parameters ---
LSTM_UNITS = [64, 32]  # Number of units in each LSTM layer
DROPOUT_RATE = 0.2     # Dropout rate for regularization
DENSE_UNITS = 32       # Number of units in dense layer
LR_REDUCTION_PATIENCE = 5  # Patience for learning rate reduction
MIN_LEARNING_RATE = 1e-6   # Minimum learning rate

# --- Data Split Parameters ---
TEST_SPLIT_RATIO = 0.2      # Test set ratio
VALIDATION_SPLIT_RATIO = 0.2 # Validation set ratio

# --- Feature Engineering Parameters ---
LAG_FEATURES = ["temperature", "humidity", "pressure", "wind_speed", "rain_1h", 
                "sea_level_m", "local_water_level_m", "satellite_rainfall_mm_hr"]
ROLLING_WINDOWS = [3, 6, 12]  # Rolling window sizes in hours
RAINFALL_COLUMNS = ["rain_1h", "satellite_rainfall_mm_hr"]
WATER_LEVEL_COLUMNS = ["sea_level_m", "local_water_level_m"]
TARGET_AREA_ELEVATION_M = 1.5  # Elevation of target area in meters

# --- Interaction Features ---
INTERACTION_FEATURES_PAIRS = [
    ("rain_1h", "satellite_rainfall_mm_hr", "rain_interaction"),
    ("humidity", "temperature", "humidity_temp_interaction"),
    ("pressure", "wind_speed", "pressure_wind_interaction")
]

# --- Target Variables ---
TARGET_RAW_WATER_LEVEL_COLUMN = "local_water_level_m"
TARGET_CLASSIFICATION = "is_flood_next_1h"
TARGET_REGRESSION = "water_level_next_1h"
ALL_TARGET_COLUMNS = [TARGET_CLASSIFICATION, TARGET_REGRESSION]

# --- Model Types to Train ---
MODEL_TYPES_TO_TRAIN = ["classification"]  # Can be ["classification", "regression"] or just one

# --- Deployment Parameters (สำหรับ FastAPI) ---
# [แก้ไข] เปลี่ยนจาก Flask API เป็น FastAPI เพื่อให้สอดคล้องกับ src/prediction_service.py
API_HOST = "0.0.0.0" 
API_PORT = 8000 # [แก้ไข] Port มาตรฐานสำหรับ FastAPI มักใช้ 8000 หรือ 8001 (เดิม 5000)
