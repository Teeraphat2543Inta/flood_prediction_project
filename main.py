# main.py

import pandas as pd
import os
import logging # [เพิ่ม] สำหรับการทำ Logging
from config import config
from src.data_ingestion import collect_historical_data
from src.model_training import train_and_evaluate_pipeline

# [เพิ่ม] ตั้งค่า Logging พื้นฐาน
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    ฟังก์ชันหลักสำหรับรัน Pipeline การฝึกโมเดลทั้งหมด:
    1. รวบรวมข้อมูลประวัติศาสตร์ (รวมข้อมูลดาวเทียม)
    2. ฝึกและประเมินโมเดลตามประเภทที่กำหนดใน config
    """
    logger.info("--- Starting Flood Prediction Model Training Pipeline ---")

    # 1. รวบรวมข้อมูลประวัติศาสตร์
    # นี่จะดึงข้อมูลจาก OWM, GEE, และแหล่งอื่นๆ (หรือข้อมูลจำลอง)
    # และรวมเข้ากับ historical_weather_data.csv หากมี
    try:
        historical_df = collect_historical_data()
        # return 
    except Exception as e:
        logger.critical(f"ERROR: Historical data collection failed: {e}. Cannot proceed with training.", exc_info=True)
        return

    if historical_df.empty:
        logger.critical("ERROR: Historical data collection resulted in an empty DataFrame. Cannot proceed with training.")
        return

    # 2. ฝึกและประเมินโมเดล
    # [แก้ไข] ใช้ config.MODEL_TYPES_TO_TRAIN เพื่อวนลูปฝึกหลายโมเดล
    for model_type_to_train in config.MODEL_TYPES_TO_TRAIN:
        logger.info(f"\n--- Training {model_type_to_train.capitalize()} Model ---")
        
        # [แก้ไข] train_and_evaluate_pipeline ไม่ต้องการ target_column แล้ว
        # เพราะมันจะเลือกเองตาม model_type_to_train ที่ส่งเข้าไป
        model, eval_results = train_and_evaluate_pipeline(
            historical_df.copy(), # [แก้ไข] ส่งสำเนาของ DataFrame ไปยัง pipeline
            model_type=model_type_to_train
        )

        if model:
            logger.info(f"{model_type_to_train.capitalize()} Model Training Complete. Evaluation Results:")
            logger.info(eval_results)
            
            # [แก้ไข] บันทึกโมเดล, scaler, และ feature_order แยกตามประเภทโมเดล
            # เพื่อให้สามารถมีโมเดล Classification และ Regression ได้พร้อมกัน
            # และเลือกใช้ใน prediction_service ได้
            
            # ต้องมั่นใจว่า path ใน config.py รองรับการแยกประเภทโมเดล
            # เช่น config.CLASSIFICATION_MODEL_SAVE_PATH, config.REGRESSION_MODEL_SAVE_PATH
            
            # ในที่นี้ เพื่อความง่าย เราจะบันทึก path ล่าสุด
            # ใน prediction_service คุณต้องมีกลไกในการระบุว่าต้องการโหลดโมเดลประเภทไหน
            # หรือจะ train เพียงโมเดลเดียวต่อการรัน main.py
            
            # หากต้องการแยกโมเดลอย่างสมบูรณ์ ควรเปลี่ยนชื่อไฟล์ที่บันทึก
            # เช่น model_classification.h5, scaler_classification.joblib
            # และมี config path สำหรับแต่ละประเภทโมเดล
            
            # สำหรับตอนนี้ ถ้าเราจะใช้ ModelCheckpoint ใน model_training
            # มันจะบันทึกตาม config.MODEL_SAVE_PATH อยู่แล้ว
            # แต่ถ้าเราต้องการให้ prediction_service เลือก type ของโมเดลได้
            # เราอาจต้องเซฟ path แยกกัน หรือมี flag ใน config ว่าจะใช้โมเดลประเภทไหน
            
            # สมมติว่าตอนนี้ เราจะฝึกโมเดลเดียวต่อการรัน main.py หรือว่า
            # config.MODEL_SAVE_PATH จะถูก update ด้วยโมเดลล่าสุดที่ฝึก
            # หากต้องการฝึกทั้ง 2 โมเดลและเก็บทั้ง 2 ควรมีการจัดการ path ที่ซับซ้อนขึ้น
            
            # [แนะนำ] เพื่อรองรับการฝึกหลายโมเดล
            # สิ่งที่คุณอาจต้องทำใน config.py:
            # CLASSIFICATION_MODEL_SAVE_PATH = os.path.join('models', 'classification_model.h5')
            # CLASSIFICATION_SCALER_SAVE_PATH = os.path.join('models', 'classification_scaler.joblib')
            # CLASSIFICATION_FEATURES_ORDER_SAVE_PATH = os.path.join('models', 'classification_features_order.joblib')
            # REGRESSION_MODEL_SAVE_PATH = os.path.join('models', 'regression_model.h5')
            # REGRESSION_SCALER_SAVE_PATH = os.path.join('models', 'regression_scaler.joblib')
            # REGRESSION_FEATURES_ORDER_SAVE_PATH = os.path.join('models', 'regression_features_order.joblib')
            
            # จากนั้นใน train_model และ preprocess_data คุณจะต้องส่ง
            # model_save_path, scaler_save_path, features_order_save_path เข้าไป
            # ซึ่งจะทำให้โค้ดยาวขึ้น แต่ยืดหยุ่นกว่า
            
            # สำหรับโค้ดปัจจุบันใน main.py และสมมติฐานว่า config.py จะมีแค่ path เดียว
            # ให้แสดง path ที่ถูกบันทึกล่าสุด
            logger.info(f"Trained {model_type_to_train.capitalize()} model saved to: {config.MODEL_SAVE_PATH}")
            logger.info(f"Scaler for {model_type_to_train.capitalize()} model saved to: {config.SCALER_SAVE_PATH}")
            logger.info(f"Feature order for {model_type_to_train.capitalize()} model saved to: {config.FEATURES_ORDER_SAVE_PATH}")
        else:
            logger.error(f"Failed to train {model_type_to_train.capitalize()} model.")

    logger.info("\n--- Flood Prediction Model Training Pipeline Complete ---")
    logger.info("You can now run the prediction service by executing: python src/prediction_service.py")

if __name__ == "__main__":
    main()
