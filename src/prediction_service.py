# src/prediction_service.py

from flask import Flask, jsonify, request
import datetime
import joblib # สำหรับโหลด scaler และ feature order
from tensorflow.keras.models import load_model # สำหรับโหลดโมเดล Keras
import numpy as np
import os
from pytz import timezone # [แก้ไข] Import timezone from pytz
import logging # [เพิ่ม] สำหรับการทำ Logging

from config import config # นำเข้าค่ากำหนด
from src.data_ingestion import get_realtime_data_for_prediction # ดึงข้อมูล Real-time
from src.data_preprocessing import prepare_realtime_input # เตรียม Input สำหรับโมเดล

app = Flask(__name__)

# [เพิ่ม] ตั้งค่า Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ตัวแปร Global สำหรับเก็บโมเดล, scaler, และลำดับ Feature
# จะโหลดเมื่อแอปพลิเคชัน Flask เริ่มทำงาน
model = None
scaler = None
features_order = None
# [เพิ่ม] เก็บ model_type เพื่อใช้ในการตีความผลการทำนาย
loaded_model_type = None 

def load_assets():
    """
    โหลดโมเดล, scaler, และลำดับ Feature จากไฟล์ที่บันทึกไว้.
    ควรเรียกใช้เมื่อแอปพลิเคชันเริ่มทำงาน.
    """
    global model, scaler, features_order, loaded_model_type
    try:
        # [แก้ไข] ตรวจสอบ path ของไฟล์ก่อนโหลดเพื่อ log error ที่ชัดเจนขึ้น
        if not os.path.exists(config.MODEL_SAVE_PATH):
            logger.error(f"Model file not found at {config.MODEL_SAVE_PATH}. Please train the model first.")
            return False
        if not os.path.exists(config.SCALER_SAVE_PATH):
            logger.error(f"Scaler file not found at {config.SCALER_SAVE_PATH}. Please train the model first.")
            return False
        if not os.path.exists(config.FEATURES_ORDER_SAVE_PATH):
            logger.error(f"Features order file not found at {config.FEATURES_ORDER_SAVE_PATH}. Please train the model first.")
            return False

        model = load_model(config.MODEL_SAVE_PATH)
        scaler = joblib.load(config.SCALER_SAVE_PATH)
        features_order = joblib.load(config.FEATURES_ORDER_SAVE_PATH)
        
        # [เพิ่ม] ตรวจสอบ Loss Function ของโมเดลเพื่อกำหนด loaded_model_type
        if model.loss == 'binary_crossentropy':
            loaded_model_type = 'classification'
        elif model.loss == 'mean_squared_error': # หรือ 'mse'
            loaded_model_type = 'regression'
        else:
            logger.warning(f"Unknown model loss function: {model.loss}. Defaulting to classification interpretation.")
            loaded_model_type = 'classification' # Default เพื่อป้องกัน error

        logger.info(f"Model, scaler, and features order loaded successfully. Detected model type: {loaded_model_type}")
        return True
    except Exception as e:
        logger.error(f"Error loading model assets: {e}", exc_info=True) # exc_info=True เพื่อแสดง traceback
        return False

@app.route('/predict_flood_risk', methods=['GET'])
def predict_flood():
    """
    API endpoint สำหรับพยากรณ์ความเสี่ยงน้ำท่วมสำหรับชั่วโมงถัดไป.
    - ดึงข้อมูล Real-time ล่าสุด (รวมข้อมูลดาวเทียม)
    - ประมวลผลข้อมูล (สร้าง Feature, Scale)
    - ทำนายด้วยโมเดล
    - ส่งคืนผลการทำนาย
    """
    # [แก้ไข] ตรวจสอบ Assets ในทุก Request
    if model is None or scaler is None or features_order is None or loaded_model_type is None:
        logger.warning("Model assets not fully loaded. Attempting to reload.")
        if not load_assets():
            return jsonify({"error": "Model assets not loaded. Check server logs for details."}), 500

    try:
        # 1. ดึงข้อมูล Real-time ดิบ (รวมช่วงข้อมูลย้อนหลังสำหรับ Lag Features)
        # [แก้ไข] ใช้ config.REALTIME_DATA_WINDOW_HOURS เพื่อกำหนดขนาด Window
        latest_raw_data_window_df = get_realtime_data_for_prediction(
            window_hours=config.REALTIME_DATA_WINDOW_HOURS
        )
        
        # ตรวจสอบว่ามีข้อมูลเพียงพอหรือไม่
        # latest_raw_data_window_df ควรมีข้อมูลอย่างน้อย (max_lag + max_rolling_window) + PREDICTION_HORIZON_HOURS
        # เพื่อให้ create_features สร้าง lag features และ rolling stats ได้ครบถ้วนสำหรับแถวสุดท้าย
        
        # คำนวณ minimum_required_rows โดยอิงจาก config.SEQUENCE_LENGTH 
        # และขนาด window ที่ใช้ในการคำนวณ rolling stats ที่ใหญ่ที่สุด (สมมติว่าเป็น max(config.ROLLING_WINDOWS))
        # และ + PREDICTION_HORIZON_HOURS สำหรับ target
        
        # [แก้ไข] คำนวณ Minimum Required Rows ให้ถูกต้อง
        max_lag = config.SEQUENCE_LENGTH
        # หา rolling window ที่ใหญ่ที่สุด
        max_rolling_window = max(config.ROLLING_WINDOWS) if config.ROLLING_WINDOWS else 0
        
        # ข้อมูลดิบต้องมีอย่างน้อย max(max_lag, max_rolling_window) + PREDICTION_HORIZON_HOURS
        # และถ้าเกิดการ dropna ใน create_features มันจะลบ row แรกๆ ออก
        # ดังนั้นควรให้ window_hours มากกว่าค่านี้เล็กน้อย
        
        # หาก latest_raw_data_window_df คือข้อมูลที่จำเป็นสำหรับสร้าง feature ที่ **ล่าสุด**
        # จำนวนแถวที่ควรมีคือ max(lag_window, rolling_window) + 1 (สำหรับปัจจุบัน)
        # และต้องมีข้อมูลเผื่อการ shift target variable ด้วย PREDICTION_HORIZON_HOURS
        # (แต่ target variable ไม่ได้ถูกใช้ในการ predict_flood_risk)
        # ดังนั้น เราต้องการข้อมูลเพียงพอสำหรับสร้าง features_order สำหรับ *จุดเวลาล่าสุด*
        # ซึ่ง features_order มี lags และ rolling stats รวมอยู่แล้ว
        # จำนวน row ที่ต้องการคือ max(lag_value) + max(rolling_window) + 1
        
        # เนื่องจาก prepare_realtime_input จะเรียก create_features และ dropna
        # เราต้องแน่ใจว่า raw data window มีข้อมูลมากพอที่หลังจาก dropna แล้วจะเหลืออย่างน้อย 1 แถวสำหรับทำนาย
        # โดยทั่วไปคือ Max of (Largest Lag, Largest Rolling Window) + PREDICTION_HORIZON_HOURS
        minimum_required_rows = max(max_lag, max_rolling_window) + config.PREDICTION_HORIZON_HOURS # +1 if target is not shifted inside 
        
        if latest_raw_data_window_df.empty or len(latest_raw_data_window_df) < minimum_required_rows:
            logger.warning(f"Not enough real-time data in window. Need at least {minimum_required_rows} hours, got {len(latest_raw_data_window_df)}.")
            return jsonify({"error": f"Not enough real-time data in window. Need at least {minimum_required_rows} hours, got {len(latest_raw_data_window_df)}. Data ingestion window may be too small or data is missing."}), 500

        # 2. เตรียม Input (สร้าง Feature และปรับขนาด)
        try:
            X_input_for_model = prepare_realtime_input(latest_raw_data_window_df.copy(), scaler, features_order)
        except ValueError as ve:
            logger.error(f"Error preparing real-time input: {ve}", exc_info=True)
            return jsonify({"error": f"Failed to prepare input data: {ve}"}), 500
        except Exception as e:
            logger.error(f"Unexpected error during input preparation: {e}", exc_info=True)
            return jsonify({"error": f"An unexpected error occurred during input preparation: {str(e)}"}), 500

        # 3. ทำนายผลด้วยโมเดล
        prediction_output = model.predict(X_input_for_model)[0][0] # ดึงค่าออกมาจาก array

        # 4. ตีความผลการทำนายตาม loaded_model_type
        if loaded_model_type == 'classification':
            is_flood = (prediction_output > config.CLASSIFICATION_THRESHOLD).astype(int) # [แก้ไข] ใช้ config.CLASSIFICATION_THRESHOLD
            certainty = prediction_output if is_flood else (1 - prediction_output)
            
            message = "Flood predicted for next hour." if is_flood else "No flood predicted for next hour."
            
            response = {
                "status": "success",
                "predicted_is_flood": bool(is_flood), 
                "prediction_score": float(prediction_output), 
                "certainty": float(certainty), 
                "message": message,
                "model_type": "classification",
                "prediction_time_utc": datetime.datetime.now(timezone('UTC')).isoformat(), # [เพิ่ม] ใช้ UTC สำหรับ Timestamp
                "prediction_time_local": datetime.datetime.now(timezone(config.TIMEZONE)).isoformat()
            }
            logger.info(f"Classification prediction: {response}")
            return jsonify(response)

        elif loaded_model_type == 'regression':
            predicted_water_level = prediction_output
            is_flood_threshold_exceeded = (predicted_water_level > config.FLOOD_WATER_LEVEL_THRESHOLD_M)
            
            message = "High water level predicted for next hour." if is_flood_threshold_exceeded else "Normal water level predicted for next hour."
            
            response = {
                "status": "success",
                "predicted_water_level_m": float(predicted_water_level), 
                "flood_threshold_m": config.FLOOD_WATER_LEVEL_THRESHOLD_M,
                "threshold_exceeded": bool(is_flood_threshold_exceeded), 
                "message": message,
                "model_type": "regression",
                "prediction_time_utc": datetime.datetime.now(timezone('UTC')).isoformat(), # [เพิ่ม] ใช้ UTC สำหรับ Timestamp
                "prediction_time_local": datetime.datetime.now(timezone(config.TIMEZONE)).isoformat()
            }
            logger.info(f"Regression prediction: {response}")
            return jsonify(response)
        else:
            logger.error(f"Invalid or unhandled loaded_model_type: {loaded_model_type}. Cannot interpret prediction.")
            return jsonify({"error": f"Server error: Invalid model type '{loaded_model_type}' for interpretation."}), 500

    except Exception as e:
        logger.error(f"Unhandled prediction error: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred during prediction: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
def status():
    """API endpoint สำหรับตรวจสอบสถานะของบริการ."""
    status_info = {
        "status": "Service is running",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "features_order_loaded": features_order is not None,
        "loaded_model_type": loaded_model_type,
        "last_checked_utc": datetime.datetime.now(timezone('UTC')).isoformat(), # [เพิ่ม] ใช้ UTC
        "last_checked_local": datetime.datetime.now(timezone(config.TIMEZONE)).isoformat()
    }
    logger.info(f"Status check: {status_info}")
    return jsonify(status_info)

# รันเมื่อไฟล์นี้ถูกเรียกใช้โดยตรง
if __name__ == '__main__':
    # โหลด Assets เมื่อ Flask app เริ่มทำงาน
    # [แก้ไข] ใช้ logger แทน print
    logger.info("Starting Flask prediction service...")
    if load_assets():
        # ใน Production ควรใช้ Gunicorn หรือ Waitress แทน app.run(debug=True)
        # debug=True ไม่เหมาะสำหรับ Production เพราะอาจมีช่องโหว่ด้านความปลอดภัยและประสิทธิภาพ
        logger.info(f"Flask app running on {config.API_HOST}:{config.API_PORT}")
        app.run(host=config.API_HOST, port=config.API_PORT, debug=config.FLASK_DEBUG_MODE, use_reloader=False) # [แก้ไข] Use_reloader=False ใน Production
    else:
        logger.critical("Failed to load model assets. Exiting prediction service.")
        # [เพิ่ม] อาจจะ raise Exception หรือ sys.exit() ถ้าจำเป็น
        # import sys
        # sys.exit(1)