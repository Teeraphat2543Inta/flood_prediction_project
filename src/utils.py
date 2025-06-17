# src/utils.py

import pandas as pd
import numpy as np
import datetime
from pytz import timezone, UTC # [แก้ไข] Import UTC เพื่อความชัดเจน

# นำเข้า config เพื่อใช้ TIMEZONE
from config import config 

def calculate_lunar_phase(date_time_obj):
    """
    คำนวณข้างขึ้นข้างแรม (0-7, โดยที่ 0=เดือนดับ, 4=เดือนเต็มดวง)
    เป็นการประมาณค่าอย่างง่ายตามจำนวนวันนับตั้งแต่จุดอ้างอิง.
    
    Parameters:
    - date_time_obj (datetime.datetime): วัตถุ datetime ที่มี timezone หรือไม่ก็ได้.
                                          หากไม่มี timezone จะถูกสันนิษฐานว่าเป็น timezone ท้องถิ่น (จาก config.TIMEZONE)
                                          แล้วแปลงเป็น UTC ก่อนการคำนวณ.
    
    Returns:
    - float: ดัชนีข้างขึ้นข้างแรม (0-7), โดยที่ 0 คือเดือนดับ, 4 คือเดือนเต็มดวง.
    """
    # กำหนดวันที่เดือนดับ (New Moon) ที่เป็นที่รู้จักเป็นจุดอ้างอิง
    # สำหรับการประมาณค่า เราสามารถใช้วันที่ใดก็ได้เป็นจุดเริ่มต้นของวงจรจันทรคติ
    # เช่น 2000-01-06 เที่ยง UTC ซึ่งเป็นวันที่เดือนดับ
    # [แก้ไข] กำหนด reference_date โดยใช้ pytz.UTC โดยตรงเพื่อความสอดคล้อง
    reference_date = datetime.datetime(2000, 1, 6, 12, 0, 0).replace(tzinfo=UTC) 

    # ตรวจสอบและแปลง input date_time_obj ให้อยู่ใน UTC เพื่อความสอดคล้องในการคำนวณ
    if date_time_obj.tzinfo is None:
        # ถ้าไม่มี timezone ให้สันนิษฐานว่าเป็น timezone ท้องถิ่นแล้วแปลงเป็น UTC
        # [แก้ไข] ใช้ .localize() และ .astimezone(UTC)
        try:
            local_tz = timezone(config.TIMEZONE)
            date_time_obj_utc = local_tz.localize(date_time_obj).astimezone(UTC)
        except Exception as e:
            # กรณีที่ localize ไม่ได้ อาจจะเพราะ datetime นั้นไม่ถูกต้อง
            print(f"Warning: Could not localize datetime object {date_time_obj}. Assuming it's UTC. Error: {e}")
            date_time_obj_utc = date_time_obj.replace(tzinfo=UTC)
    else:
        # ถ้ามี timezone อยู่แล้ว ให้แปลงเป็น UTC
        date_time_obj_utc = date_time_obj.astimezone(UTC)

    # คำนวณจำนวนวันทั้งหมดที่ต่างจากวันที่อ้างอิง
    # total_seconds() จะให้ผลลัพธ์เป็นวินาที, หารด้วยจำนวนวินาทีในหนึ่งวัน
    diff_seconds = (date_time_obj_utc - reference_date).total_seconds()
    
    # วงจรข้างขึ้นข้างแรมโดยประมาณคือ 29.530588 วัน (Synodic Month)
    synodic_month_seconds = 29.530588 * 24 * 3600
    
    # คำนวณดัชนีข้างขึ้นข้างแรม (0-7) โดยแบ่งวงจรออกเป็น 8 เฟส
    lunar_phase_index = (diff_seconds % synodic_month_seconds) / synodic_month_seconds * 8
    
    return lunar_phase_index

def datetime_to_unix(dt):
    """
    แปลงวัตถุ datetime เป็น Unix timestamp (จำนวนวินาทีนับตั้งแต่ Epoch UTC).
    
    Parameters:
    - dt (datetime.datetime): วัตถุ datetime ที่มี timezone หรือไม่ก็ได้.
                              หากไม่มี timezone จะถูกสันนิษฐานว่าเป็น UTC.
    
    Returns:
    - int: Unix timestamp.
    """
    # [แก้ไข] หากไม่มี timezone ให้ถือว่าเป็น UTC โดยปริยาย
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC) # [แก้ไข] ใช้ UTC จาก pytz
    
    return int(dt.timestamp())

def unix_to_datetime(ts, tz=config.TIMEZONE):
    """
    แปลง Unix timestamp เป็นวัตถุ datetime ใน timezone ที่กำหนด.
    
    Parameters:
    - ts (int/float): Unix timestamp.
    - tz (str): ชื่อ timezone (เช่น 'Asia/Bangkok'). Default ใช้ config.TIMEZONE.
    
    Returns:
    - datetime.datetime: วัตถุ datetime ที่มี timezone.
    """
    # [แก้ไข] ใช้ timezone.localize() หรือ astimezone() เพื่อจัดการ timezone ที่ถูกต้อง
    target_tz = timezone(tz)
    # datetime.datetime.fromtimestamp จะคืนค่าเป็น local time หากไม่มี tz
    # หรือคืนค่าใน tz ที่ระบุหากระบุ tz
    return datetime.datetime.fromtimestamp(ts, tz=target_tz)


# --- ฟังก์ชันยูทิลิตี้เพิ่มเติมที่อาจเป็นประโยชน์ ---

def check_and_fill_missing_values(df, strategy='ffill', columns=None):
    """
    ตรวจสอบและเติมค่าที่หายไปใน DataFrame.
    
    Parameters:
    - df (pd.DataFrame): DataFrame ที่ต้องการจัดการค่า Missing.
    - strategy (str): กลยุทธ์ในการเติมค่า ('ffill', 'bfill', 'mean', 'median', 'zero').
    - columns (list): รายชื่อคอลัมน์ที่ต้องการเติมค่า ถ้า None จะเติมทุกคอลัมน์.
    
    Returns:
    - pd.DataFrame: DataFrame ที่เติมค่า Missing แล้ว.
    """
    df_copy = df.copy()
    cols_to_fill = columns if columns is not None else df_copy.columns

    print(f"Checking for missing values. Original NaNs: {df_copy.isnull().sum().sum()}")

    for col in cols_to_fill:
        if col not in df_copy.columns:
            print(f"Warning: Column '{col}' not found in DataFrame for missing value handling.")
            continue

        if df_copy[col].isnull().any():
            if strategy == 'ffill':
                df_copy[col].fillna(method='ffill', inplace=True)
            elif strategy == 'bfill':
                df_copy[col].fillna(method='bfill', inplace=True)
            elif strategy == 'mean':
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'median':
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'zero':
                df_copy[col].fillna(0, inplace=True)
            else:
                print(f"Warning: Unknown fill strategy '{strategy}'. Skipping column '{col}'.")
    
    print(f"NaNs after filling: {df_copy.isnull().sum().sum()}")
    return df_copy

def detect_outliers_iqr(series, k=1.5):
    """
    ตรวจจับ Outliers โดยใช้ IQR Method.
    
    Parameters:
    - series (pd.Series): Series ที่ต้องการตรวจจับ Outliers.
    - k (float): ตัวคูณสำหรับ IQR (ทั่วไปคือ 1.5).
    
    Returns:
    - pd.Series: Boolean Series ที่เป็น True สำหรับ Outliers.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    
    return (series < lower_bound) | (series > upper_bound)

# ตัวอย่างการใช้งาน (สามารถเอาออกเมื่อนำไปใช้จริง)
if __name__ == '__main__':
    # ทดสอบ calculate_lunar_phase
    print("--- Testing calculate_lunar_phase ---")
    # Datetime ไม่มี timezone
    dt_no_tz = datetime.datetime(2023, 10, 28, 18, 0, 0) 
    lunar_phase_no_tz = calculate_lunar_phase(dt_no_tz)
    print(f"Lunar phase for {dt_no_tz} (no tz): {lunar_phase_no_tz:.2f}")

    # Datetime มี timezone (UTC)
    dt_utc = datetime.datetime(2023, 10, 28, 11, 0, 0, tzinfo=timezone('UTC')) # เวลาเดียวกับด้านบนใน UTC
    lunar_phase_utc = calculate_lunar_phase(dt_utc)
    print(f"Lunar phase for {dt_utc} (UTC): {lunar_phase_utc:.2f}")

    # Datetime มี timezone (Bangkok)
    dt_bkk = datetime.datetime(2023, 10, 28, 18, 0, 0, tzinfo=timezone('Asia/Bangkok'))
    lunar_phase_bkk = calculate_lunar_phase(dt_bkk)
    print(f"Lunar phase for {dt_bkk} (BKK): {lunar_phase_bkk:.2f}")
    
    print("\n--- Testing datetime_to_unix and unix_to_datetime ---")
    now_utc = datetime.datetime.now(UTC)
    unix_ts = datetime_to_unix(now_utc)
    print(f"Current UTC datetime: {now_utc}")
    print(f"Unix timestamp: {unix_ts}")
    
    dt_from_unix_bkk = unix_to_datetime(unix_ts, tz='Asia/Bangkok')
    print(f"Datetime from Unix (Asia/Bangkok): {dt_from_unix_bkk}")
    
    dt_from_unix_utc = unix_to_datetime(unix_ts, tz='UTC')
    print(f"Datetime from Unix (UTC): {dt_from_unix_utc}")

    # ทดสอบฟังก์ชันใหม่
    print("\n--- Testing check_and_fill_missing_values ---")
    data = {'col1': [1, 2, np.nan, 4, 5], 
            'col2': [np.nan, 20, 30, np.nan, 50],
            'col3': [100, 110, 120, 130, 140]}
    df_test = pd.DataFrame(data)
    print("Original DataFrame:\n", df_test)

    df_filled_ffill = check_and_fill_missing_values(df_test, strategy='ffill')
    print("\nFilled with ffill:\n", df_filled_ffill)

    df_filled_mean = check_and_fill_missing_values(df_test, strategy='mean', columns=['col2'])
    print("\nFilled col2 with mean:\n", df_filled_mean)

    print("\n--- Testing detect_outliers_iqr ---")
    data_outliers = pd.Series([1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10])
    outliers = detect_outliers_iqr(data_outliers)
    print(f"Data for outlier detection: {data_outliers.tolist()}")
    print(f"Outliers detected: {data_outliers[outliers].tolist()}")