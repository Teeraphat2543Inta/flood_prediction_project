# src/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib # สำหรับบันทึก/โหลด scaler
import os

from config import config # นำเข้าค่ากำหนด
from src.utils import calculate_lunar_phase # นำเข้าฟังก์ชันยูทิลิตี้

def create_features(df):
    """
    สร้างคุณลักษณะ (Features) ใหม่จากข้อมูลดิบ.
    นี่เป็นขั้นตอนที่สำคัญมากสำหรับประสิทธิภาพของโมเดล Time Series.
    รวมถึงการสร้าง Lag Features, Rolling Statistics และ Time-based Features.
    
    Parameters:
    - df (pd.DataFrame): DataFrame ที่มีข้อมูลดิบ (ต้องมี Timestamp เป็น Index)
    
    Returns:
    - pd.DataFrame: DataFrame ที่มี Feature เพิ่มเติม และลบแถวที่มี NaN จากการสร้าง Feature
    """
    df_copy = df.copy() # ใช้ชื่อ df_copy เพื่อไม่ให้ทับ df เดิม
    
    print("Creating features...")
    
    # --- Time-based Features ---
    # ช่วยให้โมเดลเรียนรู้รูปแบบตามช่วงเวลา
    df_copy['hour_of_day'] = df_copy.index.hour
    df_copy['day_of_week'] = df_copy.index.dayofweek
    df_copy['month'] = df_copy.index.month
    df_copy['day_of_year'] = df_copy.index.dayofyear
    df_copy['week_of_year'] = df_copy.index.isocalendar().week.astype(int) 
    df_copy['is_weekend'] = (df_copy.index.dayofweek >= 5).astype(int) 
    df_copy['quarter'] = df_copy.index.quarter 
    
    # --- Lunar Phase Feature ---
    df_copy['lunar_phase'] = df_copy.index.map(calculate_lunar_phase)

    # --- Lag Features (คุณลักษณะย้อนหลัง) ---
    # [แก้ไข] ใช้ config.LAG_FEATURES แทนการ hardcode
    features_to_lag = config.LAG_FEATURES 
    
    for feature in features_to_lag:
        if feature in df_copy.columns:
            for i in range(1, config.SEQUENCE_LENGTH + 1):
                df_copy[f'{feature}_lag_{i}h'] = df_copy[feature].shift(i)
        else:
            print(f"Warning: Feature '{feature}' (for lagging) not found in DataFrame. Creating as NaNs.")
            for i in range(1, config.SEQUENCE_LENGTH + 1):
                df_copy[f'{feature}_lag_{i}h'] = np.nan 

    # --- Rolling Statistics (ค่าสถิติเคลื่อนที่) ---
    # [แก้ไข] ใช้ config.ROLLING_WINDOWS
    rolling_windows = config.ROLLING_WINDOWS # เช่น [3, 6, 12]

    # Rolling sum ของปริมาณน้ำฝน
    # [แก้ไข] ใช้ config.RAINFALL_COLUMNS แทนการ hardcode 'rain_1h'
    for rain_col in config.RAINFALL_COLUMNS: 
        if rain_col in df_copy.columns:
            for window in rolling_windows:
                df_copy[f'{rain_col}_sum_{window}h'] = df_copy[rain_col].rolling(window=window, min_periods=1).sum().shift(1)
        else:
            for window in rolling_windows:
                df_copy[f'{rain_col}_sum_{window}h'] = 0.0

    # Rolling average ของระดับน้ำ
    # [แก้ไข] ใช้ config.WATER_LEVEL_COLUMNS แทนการ hardcode 'water_level_m'
    for water_col in config.WATER_LEVEL_COLUMNS:
        if water_col in df_copy.columns:
            for window in rolling_windows:
                df_copy[f'{water_col}_avg_{window}h'] = df_copy[water_col].rolling(window=window, min_periods=1).mean().shift(1)
        else:
            for window in rolling_windows:
                df_copy[f'{water_col}_avg_{window}h'] = 0.0

    # Rolling std deviation ของระดับน้ำ
    for water_col in config.WATER_LEVEL_COLUMNS:
        if water_col in df_copy.columns:
            for window in rolling_windows:
                df_copy[f'{water_col}_std_{window}h'] = df_copy[water_col].rolling(window=window, min_periods=1).std().shift(1)
        else:
            for window in rolling_windows:
                df_copy[f'{water_col}_std_{window}h'] = 0.0

    # --- Static Features (คุณลักษณะคงที่สำหรับพื้นที่) ---
    # [แก้ไข] ใช้ config.TARGET_AREA_ELEVATION_M
    df_copy['elevation_m'] = config.TARGET_AREA_ELEVATION_M 

    # --- Interaction Features (คุณลักษณะเชิงปฏิสัมพันธ์) ---
    # [แก้ไข] ใช้ config.INTERACTION_FEATURES_PAIRS
    for pair in config.INTERACTION_FEATURES_PAIRS:
        feat1, feat2, name = pair[0], pair[1], pair[2]
        if feat1 in df_copy.columns and feat2 in df_copy.columns:
            df_copy[name] = df_copy[feat1] * df_copy[feat2]
        else:
            df_copy[name] = 0.0

    # --- Target Variable ---
    # [แก้ไข] ใช้ config.TARGET และ config.FLOOD_WATER_LEVEL_THRESHOLD_M
    if config.TARGET_RAW_WATER_LEVEL_COLUMN in df_copy.columns: # ตรวจสอบว่าคอลัมน์ระดับน้ำดิบมีอยู่จริง
        # Target สำหรับ Regression
        df_copy[config.TARGET_REGRESSION] = df_copy[config.TARGET_RAW_WATER_LEVEL_COLUMN].shift(-config.PREDICTION_HORIZON_HOURS)
        # Target สำหรับ Classification
        df_copy[config.TARGET_CLASSIFICATION] = (df_copy[config.TARGET_RAW_WATER_LEVEL_COLUMN].shift(-config.PREDICTION_HORIZON_HOURS) > config.FLOOD_WATER_LEVEL_THRESHOLD_M).astype(int)
    else:
        print(f"Warning: '{config.TARGET_RAW_WATER_LEVEL_COLUMN}' not found. Cannot create target variables.")
        df_copy[config.TARGET_REGRESSION] = np.nan
        df_copy[config.TARGET_CLASSIFICATION] = np.nan

    # --- จัดการค่า NaN ที่เกิดจากการสร้าง Lag/Rolling/Target ---
    initial_rows = df_copy.shape[0]
    # [แก้ไข] เพิ่มการกรองก่อน dropna เพื่อให้แน่ใจว่า drop เฉพาะที่เกี่ยวข้องกับ feature creation
    # ควร dropna ด้วย subset ของคอลัมน์ที่เป็น feature + target เท่านั้น
    
    # Identify columns that are expected to have NaNs from lagging/rolling/shifting
    columns_with_expected_nans = [col for col in df_copy.columns if any(s in col for s in ['_lag_', '_sum_', '_avg_', '_std_'])]
    columns_with_expected_nans.extend([config.TARGET_REGRESSION, config.TARGET_CLASSIFICATION])
    
    # Filter for columns that actually exist in the DataFrame
    existing_expected_nans_cols = [col for col in columns_with_expected_nans if col in df_copy.columns]

    if not existing_expected_nans_cols:
        print("No columns identified for targeted NaN removal. Skipping specific dropna.")
    else:
        df_copy.dropna(subset=existing_expected_nans_cols, inplace=True)

    rows_dropped = initial_rows - df_copy.shape[0]
    print(f"Dropped {rows_dropped} rows due to NaN values after feature creation (mostly from lags/rolling/target).")
    
    print(f"Features created. New shape: {df_copy.shape}")
    return df_copy

def preprocess_data(df, is_training=True, target_column=config.TARGET_CLASSIFICATION): # [แก้ไข] เพิ่ม is_training และใช้ config.TARGET_CLASSIFICATION เป็น default
    """
    ทำความสะอาด, แปลง, และปรับขนาด (Scale) ข้อมูลสำหรับโมเดล.
    
    Parameters:
    - df (pd.DataFrame): DataFrame ที่มี Feature แล้ว
    - is_training (bool): True ถ้าเป็นการฝึกโมเดล (จะ fit scaler และบันทึก) False ถ้าเป็นการทำนาย (จะโหลด scaler)
    - target_column (str): ชื่อคอลัมน์ของ Target Variable ('is_flood_next_1h' หรือ 'water_level_next_1h')
    
    Returns:
    - X_scaled_df (pd.DataFrame): Features ที่ถูกปรับขนาดแล้ว
    - y (pd.Series): Target Variable (เฉพาะเมื่อ is_training=True)
    - scaler (MinMaxScaler): Scaler ที่ใช้ในการปรับขนาด
    """
    print(f"Preprocessing data for target: {target_column} (is_training={is_training})...")
    
    # ตรวจสอบว่า df มีข้อมูลหรือไม่
    if df.empty:
        raise ValueError("Input DataFrame is empty for preprocessing.")

    # [แก้ไข] สร้าง Features จาก df
    df_with_features = create_features(df)

    # แยก Features (X) และ Target (y)
    # [แก้ไข] ใช้ config.ALL_TARGET_COLUMNS เพื่อระบุคอลัมน์ Target ทั้งหมด
    all_target_columns = config.ALL_TARGET_COLUMNS 
    
    feature_columns = [col for col in df_with_features.columns if col not in all_target_columns]
    
    X = df_with_features[feature_columns]
    y = df_with_features[target_column] # y จะถูกใช้แค่ในโหมด training
    
    # ตรวจสอบว่า X มีข้อมูลหรือไม่
    if X.empty:
        raise ValueError("Feature DataFrame (X) is empty after feature creation and NaN dropping. Check input data.")
    
    # โหลดหรือสร้าง Scaler
    scaler = None
    if is_training:
        print("Fitting and transforming scaler...")
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # บันทึก Scaler และลำดับของ Features
        # [แก้ไข] สร้าง Directory หากยังไม่มี
        os.makedirs(os.path.dirname(config.SCALER_SAVE_PATH), exist_ok=True)
        joblib.dump(scaler, config.SCALER_SAVE_PATH)
        joblib.dump(X.columns.tolist(), config.FEATURES_ORDER_SAVE_PATH)
        print(f"Scaler and feature order saved to {config.SCALER_SAVE_PATH} and {config.FEATURES_ORDER_SAVE_PATH}")
    else: # For prediction
        print("Loading existing scaler and feature order...")
        try:
            scaler = joblib.load(config.SCALER_SAVE_PATH)
            loaded_features_order = joblib.load(config.FEATURES_ORDER_SAVE_PATH)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Scaler or Feature Order file not found. Please train the model first. Error: {e}")
        
        # [แก้ไข] ตรวจสอบและ Reindex X ให้ตรงกับลำดับ Features ที่โมเดลคาดหวัง
        # ต้องมั่นใจว่า Features ที่สร้างใน create_features_for_realtime (ด้านล่าง) มีครบถ้วน
        # และอยู่ในลำดับที่ถูกต้อง
        X_reindexed = X.reindex(columns=loaded_features_order, fill_value=0) # เติม 0 ใน Feature ที่อาจหายไป

        # ตรวจสอบว่ามี NaN หลังจาก reindex หรือไม่
        if X_reindexed.isnull().any().any():
            print("Warning: NaN values found after reindexing features. Filling with 0.")
            X_reindexed.fillna(0, inplace=True) # เติม 0 สำหรับ Feature ที่อาจถูก reindex แล้วเป็น NaN

        X_scaled = scaler.transform(X_reindexed)
        X_scaled_df = pd.DataFrame(X_scaled, columns=loaded_features_order, index=X.index)

    print(f"Data preprocessed. X_scaled shape: {X_scaled_df.shape}, y shape: {y.shape if is_training else 'N/A'}")
    return (X_scaled_df, y, scaler) if is_training else (X_scaled_df, None, scaler)


def prepare_realtime_input(latest_raw_data_window_df, scaler, features_order):
    """
    เตรียม Input สำหรับโมเดลจากข้อมูล Real-time.
    ขั้นตอน: สร้าง Feature -> เลือก Feature ที่ถูกต้อง -> Scale -> Reshape สำหรับโมเดล.
    
    Parameters:
    - latest_raw_data_window_df (pd.DataFrame): DataFrame ที่มีข้อมูลดิบ
                                               (รวมถึงข้อมูลย้อนหลังที่จำเป็นสำหรับ Lag Features)
                                               เป็น Index ด้วย Timestamp
    - scaler (MinMaxScaler): Scaler ที่โหลดมาจากตอนฝึกโมเดล
    - features_order (list): ลำดับของ Feature ที่โมเดลถูกฝึกมาด้วย
    
    Returns:
    - numpy.ndarray: Input ที่พร้อมสำหรับโมเดล (รูปทรง (1, num_timesteps, num_features) สำหรับ LSTM)
                     num_timesteps จะเป็น 1 ถ้าโมเดลรับ Lag Features เข้ามาแล้ว
                     แต่ถ้าโมเดลรับ sequence ของข้อมูลดิบ (เช่น (SEQUENCE_LENGTH, num_features_raw)),
                     ต้อง reshape ต่างออกไป
    """
    print("Preparing real-time input for prediction...")
    
    # 1. สร้าง Feature จาก Raw Data Window
    # create_features จะสร้าง Lag/Rolling Stats โดยใช้ข้อมูลในอดีตที่อยู่ใน latest_raw_data_window_df
    # [แก้ไข] ใช้ create_features โดยตรง
    processed_window_df = create_features(latest_raw_data_window_df)
    
    # [แก้ไข] ลบข้อมูล Target ออกไปก่อน
    all_target_columns = config.ALL_TARGET_COLUMNS
    processed_window_df = processed_window_df.drop(columns=[col for col in all_target_columns if col in processed_window_df.columns])


    # ตรวจสอบว่ามีข้อมูลเพียงพอหลังจากสร้าง Feature
    # เราต้องการ row ล่าสุด 1 แถว ที่มี features ครบ
    if processed_window_df.empty:
        raise ValueError("Processed data window is empty after feature creation and NaN dropping. Check input data window size.")
    
    # 2. เลือกแถวสุดท้าย ซึ่งเป็น Feature สำหรับการทำนายปัจจุบัน
    X_latest_for_prediction = processed_window_df.iloc[-1:] # ใช้ iloc[-1:] เพื่อรักษารูปแบบ DataFrame

    # [แก้ไข] Reindex X_latest_for_prediction ให้ตรงกับ features_order
    # สำคัญมากเพื่อให้ลำดับ feature ตรงกับที่โมเดลคาดหวัง
    X_latest_for_prediction = X_latest_for_prediction.reindex(columns=features_order, fill_value=0)

    # ตรวจสอบว่า X_latest_for_prediction มีค่า NaN หรือไม่
    # ในขั้นตอนนี้ไม่ควรมี NaNs แล้ว หากมีแสดงว่าการเติมค่าใน data_ingestion หรือ create_features ไม่สมบูรณ์
    if X_latest_for_prediction.isnull().any().any():
        print("Warning: NaN values found in X_latest_for_prediction after feature creation and reindexing. Filling with 0.")
        X_latest_for_prediction.fillna(0, inplace=True) # เติม 0 เป็นค่า default

    # 3. Scale Input ด้วย Scaler ที่โหลดมา
    X_latest_scaled = scaler.transform(X_latest_for_prediction)
    
    # 4. Reshape สำหรับโมเดล LSTM
    # โมเดล LSTM ต้องการ Input ในรูปแบบ (samples, timesteps, features)
    # เนื่องจากเราได้สร้าง Lag Features ใน create_features ไปแล้ว
    # แต่ละ sample จึงเป็น 1 timestep และมีจำนวน feature เท่ากับ X_latest_scaled.shape[1]
    # ถ้าโมเดลของคุณถูก train ด้วย input shape (None, num_features) เช่น Linear Layer แรกของ Keras
    # คุณอาจไม่จำเป็นต้อง reshape เป็น 3D (1, 1, num_features)
    # แต่ถ้าเป็น LSTM จริงๆ ที่รับ sequence (เช่น (batch_size, timesteps, features))
    # และคุณต้องการให้แต่ละ "timestep" เป็นข้อมูลดิบ 1 จุด
    # **แต่วิธีที่คุณสร้าง lagged features นี้ ทำให้ "timestep" = 1**
    # **ซึ่งหมายความว่า 1 แถวของข้อมูลที่ถูกสร้าง lagged features ไปแล้ว เทียบเท่ากับ 1 sequence**
    # ดังนั้น รูปแบบ (1, 1, num_features) ก็ถูกต้องแล้ว
    
    # [ข้อควรระวัง] ถ้าโมเดล LSTM ของคุณถูก Train ด้วย Input ที่แต่ละ timestep เป็นข้อมูลดิบ 
    # (เช่น คุณไม่ได้สร้าง Lagged Features ใน preprocessing แต่ใช้ Input เป็น (batch_size, SEQUENCE_LENGTH, num_raw_features))
    # ฟังก์ชันนี้จะต้องเปลี่ยนไปดึงข้อมูล raw_data_window_df เฉพาะคอลัมน์ที่ต้องการ
    # แล้ว reshape เป็น (1, SEQUENCE_LENGTH, num_raw_features)
    
    # แต่จากโค้ดของคุณในตอนนี้ ที่มีการสร้าง lagged features
    # รูปแบบ (1, 1, X_latest_scaled.shape[1]) นั้นถูกต้องแล้ว
    input_for_model = X_latest_scaled.reshape(1, 1, X_latest_scaled.shape[1]) 

    print(f"Real-time input prepared. Shape: {input_for_model.shape}")
    return input_for_model