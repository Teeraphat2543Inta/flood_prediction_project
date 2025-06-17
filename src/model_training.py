# src/model_training.py

import pandas as pd
import numpy as np
# [แก้ไข] ใช้ TimeSeriesSplit แทน train_test_split สำหรับ Time Series
from sklearn.model_selection import train_test_split, TimeSeriesSplit 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_squared_error, mean_absolute_error # [แก้ไข] เพิ่ม mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # [แก้ไข] เพิ่ม ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os

from config import config # นำเข้าค่ากำหนด
from src.data_preprocessing import create_features, preprocess_data # นำเข้าฟังก์ชัน Preprocessing

def build_model(input_shape, model_type='classification'):
    """
    สร้างและคอมไพล์โมเดล Neural Network (LSTM).
    
    Parameters:
    - input_shape (tuple): รูปร่างของ Input สำหรับโมเดล (timesteps, features)
    - model_type (str): 'classification' สำหรับทำนายว่าท่วมหรือไม่, 'regression' สำหรับทำนายระดับน้ำ
    
    Returns:
    - tf.keras.Model: โมเดล Keras ที่ถูกสร้างและคอมไพล์แล้ว
    """
    print(f"Building {model_type} model with input shape: {input_shape}...")
    model = Sequential([
        # [แก้ไข] ใช้ config.LSTM_UNITS และ config.DROPOUT_RATE
        LSTM(units=config.LSTM_UNITS[0], return_sequences=True, input_shape=input_shape), 
        Dropout(config.DROPOUT_RATE), 
        
        LSTM(units=config.LSTM_UNITS[1], return_sequences=False), 
        Dropout(config.DROPOUT_RATE),
        
        Dense(units=config.DENSE_UNITS, activation='relu'), 
        Dropout(config.DROPOUT_RATE), # [แก้ไข] เพิ่ม Dropout ตรงนี้
    ])

    if model_type == 'classification':
        model.add(Dense(units=1, activation='sigmoid')) 
        model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy', 'AUC']) 
    elif model_type == 'regression':
        model.add(Dense(units=1, activation='linear')) 
        model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), 
                      loss='mean_squared_error', 
                      metrics=['mae', 'mse']) 
    else:
        raise ValueError("Invalid model_type. Choose 'classification' or 'regression'.")

    model.summary()
    return model

def train_model(X_train, y_train, X_val, y_val, model_type='classification'):
    """
    ฝึกโมเดล.
    
    Parameters:
    - X_train (pd.DataFrame): Features สำหรับฝึก
    - y_train (pd.Series): Target สำหรับฝึก
    - X_val (pd.DataFrame): Features สำหรับตรวจสอบ (Validation)
    - y_val (pd.Series): Target สำหรับตรวจสอบ (Validation)
    - model_type (str): 'classification' หรือ 'regression'
    
    Returns:
    - tf.keras.Model: โมเดลที่ฝึกแล้ว
    - history: Object ที่เก็บประวัติการฝึก
    """
    print("Starting model training...")

    # Reshape Data สำหรับ LSTM: (samples, timesteps, features)
    # [แก้ไข] ตรวจสอบให้แน่ใจว่า X_train, X_val เป็น DataFrame ก่อน .values
    X_train_reshaped = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_val_reshaped = X_val.values.reshape(X_val.shape[0], 1, X_val.shape[1])
    
    # สร้างโมเดล
    model = build_model(input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), 
                        model_type=model_type)

    # Callbacks:
    # EarlyStopping: หยุดการฝึกเมื่อ Validation Loss ไม่ดีขึ้นเป็นระยะเวลาหนึ่ง
    early_stopping = EarlyStopping(monitor='val_loss', patience=config.PATIENCE, restore_best_weights=True)
    
    # ModelCheckpoint: บันทึกโมเดลที่ดีที่สุด (ตาม Validation Loss)
    # [แก้ไข] ตรวจสอบและสร้าง Directory สำหรับบันทึกโมเดล
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    model_checkpoint = ModelCheckpoint(config.MODEL_SAVE_PATH, 
                                       monitor='val_loss', 
                                       save_best_only=True, 
                                       mode='min', 
                                       verbose=1)

    # [เพิ่ม] ReduceLROnPlateau: ลด Learning Rate เมื่อ Validation Loss ไม่ดีขึ้น
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=config.LR_REDUCTION_PATIENCE, 
                                  min_lr=config.MIN_LEARNING_RATE, verbose=1)

    callbacks = [early_stopping, model_checkpoint, reduce_lr] # [แก้ไข] เพิ่ม reduce_lr ใน callbacks list

    # ฝึกโมเดล
    history = model.fit(
        X_train_reshaped, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(X_val_reshaped, y_val),
        callbacks=callbacks, # ใช้ callbacks list
        verbose=1
    )
    
    print("Model training complete.")
    return model, history

def evaluate_model(model, X_test, y_test, model_type='classification'):
    """
    ประเมินประสิทธิภาพของโมเดล.
    
    Parameters:
    - model (tf.keras.Model): โมเดลที่ฝึกแล้ว
    - X_test (pd.DataFrame): Features สำหรับทดสอบ
    - y_test (pd.Series): Target สำหรับทดสอบ
    - model_type (str): 'classification' หรือ 'regression'
    
    Returns:
    - dict: ผลการประเมิน
    """
    print("Evaluating model performance...")

    # [แก้ไข] ตรวจสอบให้แน่ใจว่า X_test เป็น DataFrame ก่อน .values
    X_test_reshaped = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    # ทำนายผล
    y_pred = model.predict(X_test_reshaped)

    results = {}
    if model_type == 'classification':
        y_pred_class = (y_pred > 0.5).astype(int) # แปลงความน่าจะเป็นเป็น Class (0 หรือ 1)
        
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_pred_class))
        
        print("\n--- Confusion Matrix ---")
        print(confusion_matrix(y_test, y_pred_class))
        
        # ROC AUC Score
        # [แก้ไข] ตรวจสอบจำนวนคลาสใน y_test อย่างถูกต้อง
        if len(np.unique(y_test)) > 1:
            try:
                auc_score = roc_auc_score(y_test, y_pred)
                print(f"\nROC AUC Score: {auc_score:.4f}")
                results['roc_auc_score'] = auc_score
            except ValueError as e:
                print(f"\nWarning: Could not calculate ROC AUC Score. Error: {e}")
                results['roc_auc_score'] = np.nan
        else:
            print("\nWarning: ROC AUC Score cannot be calculated as y_test contains only one class.")

        results['accuracy'] = (y_pred_class == y_test.values.reshape(-1, 1)).mean()
        results['classification_report'] = classification_report(y_test, y_pred_class, output_dict=True)
        results['confusion_matrix'] = confusion_matrix(y_test, y_pred_class).tolist()

    elif model_type == 'regression':
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred) # [แก้ไข] ใช้ sklearn.metrics.mean_absolute_error
        
        print(f"\n--- Regression Metrics ---")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")

        results['rmse'] = rmse
        results['mae'] = mae
    
    print("Model evaluation complete.")
    return results

def train_and_evaluate_pipeline(df, model_type='classification'): # [แก้ไข] ลบ target_column จาก parameter เพื่อให้ใช้ config
    """
    รัน Pipeline การฝึกและประเมินโมเดลทั้งหมด.
    
    Parameters:
    - df (pd.DataFrame): DataFrame ที่มีข้อมูลดิบที่ถูกรวบรวมแล้ว
    - model_type (str): 'classification' หรือ 'regression'
    
    Returns:
    - tf.keras.Model: โมเดลที่ฝึกแล้ว
    - evaluation_results (dict): ผลการประเมิน
    """
    print(f"\n--- Starting Full Training and Evaluation Pipeline ({model_type}) ---")
    
    # [แก้ไข] เลือก target_column ตาม model_type จาก config
    if model_type == 'classification':
        target_column = config.TARGET_CLASSIFICATION
    elif model_type == 'regression':
        target_column = config.TARGET_REGRESSION
    else:
        raise ValueError("Invalid model_type. Choose 'classification' or 'regression'.")

    # 1. Preprocess Data (สร้าง Feature และ Scaling)
    # [แก้ไข] เรียก preprocess_data โดยตรง ไม่ต้องเรียก create_features แยก
    # preprocess_data จะจัดการ create_features ภายใน
    try:
        X_scaled_df, y, scaler = preprocess_data(df.copy(), is_training=True, target_column=target_column)
    except ValueError as e:
        print(f"Error during data preprocessing: {e}")
        return None, None
    except FileNotFoundError as e:
        print(f"Error: Scaler or Feature Order files not found during preprocessing. {e}")
        return None, None

    # ตรวจสอบว่ามีข้อมูลเพียงพอหลัง Preprocessing
    if X_scaled_df.empty or y.empty:
        print("Error: Processed DataFrame (features or target) is empty. Cannot proceed with training.")
        return None, None
    
    # 2. แบ่งข้อมูล (Train, Validation, Test) สำหรับ Time Series
    # [แก้ไข] ใช้ TimeSeriesSplit เพื่อรักษาลำดับเวลา
    # ตั้งค่า Test set size ตาม config
    test_size = config.TEST_SPLIT_RATIO 
    val_size = config.VALIDATION_SPLIT_RATIO 

    # การแบ่งข้อมูลด้วย TimeSeriesSplit จะแตกต่างกันเล็กน้อย
    # เราจะแบ่งเป็น Train+Val และ Test ก่อน จากนั้นค่อยแบ่ง Train+Val อีกที

    # 1. แบ่งข้อมูลทั้งหมดเป็น Train/Validation และ Test
    # หาจุดแบ่งสำหรับ Test set
    test_split_idx = int(len(X_scaled_df) * (1 - test_size))
    X_train_val = X_scaled_df.iloc[:test_split_idx]
    y_train_val = y.iloc[:test_split_idx]
    X_test = X_scaled_df.iloc[test_split_idx:]
    y_test = y.iloc[test_split_idx:]

    # ตรวจสอบว่า test set มีข้อมูลเพียงพอ
    if X_test.empty or y_test.empty:
        print("Warning: Test set is empty after splitting. Adjust test_size or provide more data.")
        # หาก Test set ว่าง อาจต้องใช้ข้อมูลทั้งหมดเป็น Train/Val หรือพิจารณาขนาดข้อมูล
        if len(X_scaled_df) < (1 / test_size): # สมมติว่าต้องการอย่างน้อย 1 แถวใน test
            print("Not enough data for a meaningful test split. Using all data for train/val.")
            X_train_val = X_scaled_df
            y_train_val = y
            X_test = pd.DataFrame() # Make them empty
            y_test = pd.Series()
        else:
            print("Cannot proceed with an empty test set.")
            return None, None # หรือ raise Exception

    # 2. แบ่ง Train+Validation เป็น Train และ Validation
    val_split_idx = int(len(X_train_val) * (1 - val_size))
    X_train = X_train_val.iloc[:val_split_idx]
    y_train = y_train_val.iloc[:val_split_idx]
    X_val = X_train_val.iloc[val_split_idx:]
    y_val = y_train_val.iloc[val_split_idx:]

    print(f"Data split: Train={len(X_train)} rows, Validation={len(X_val)} rows, Test={len(X_test)} rows")
    
    # [แก้ไข] ตรวจสอบว่ามีข้อมูลในชุด train/val/test เพียงพอหรือไม่
    if X_train.empty or y_train.empty:
        print("Error: Training set is empty. Cannot proceed with training.")
        return None, None
    if X_val.empty or y_val.empty:
        print("Error: Validation set is empty. Cannot proceed with training. Adjust split ratios.")
        return None, None

    # 3. ฝึกโมเดล
    model, history = train_model(X_train, y_train, X_val, y_val, model_type=model_type)

    # 4. โหลดโมเดลที่ดีที่สุดกลับมา (ModelCheckpoint บันทึกไว้แล้ว)
    best_model = None
    if os.path.exists(config.MODEL_SAVE_PATH):
        try:
            best_model = load_model(config.MODEL_SAVE_PATH)
            print(f"Loaded best model from {config.MODEL_SAVE_PATH}")
        except Exception as e:
            print(f"Error loading best model: {e}. Using final trained model instead.")
            best_model = model
    else:
        best_model = model 
        print("No best model saved by ModelCheckpoint. Using final trained model.")

    # 5. ประเมินโมเดลด้วยชุด Test (ถ้า Test set ไม่ว่าง)
    evaluation_results = {}
    if not X_test.empty and not y_test.empty:
        evaluation_results = evaluate_model(best_model, X_test, y_test, model_type=model_type)
    else:
        print("Skipping final evaluation as test set is empty.")
    
    print("\n--- Full Training and Evaluation Pipeline Complete ---")
    return best_model, evaluation_results