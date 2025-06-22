# src/data_ingestion.py

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytz import timezone # ใช้ pytz เพื่อจัดการ Timezone อย่างถูกต้อง
import time
import os
import json

from config import config # นำเข้าค่ากำหนดจาก config.py
from src.utils import datetime_to_unix, unix_to_datetime, get_current_local_time # นำเข้าฟังก์ชันยูทิลิตี้ (เพิ่ม get_current_local_time)

from src.connection.connection2gee import GEEConnection

# gee = GEEConnection(json_key_path='benz-gee-50f789454788.json')
gee = GEEConnection()
ee = gee.ee

# --- ฟังก์ชันสำหรับดึงข้อมูลสภาพอากาศจาก OpenWeatherMap ---
# [แก้ไข] ปรับ history_hours ให้ใช้ config.SEQUENCE_LENGTH และ PREDICTION_HORIZON_HOURS เพื่อความยืดหยุ่น
def fetch_weather_data_from_owm(lat, lon, api_key, history_hours_needed): 
    """
    ดึงข้อมูลสภาพอากาศและการพยากรณ์ระยะสั้นจาก OpenWeatherMap API (5 วัน / 3 ชั่วโมง).
    เราจะใช้ข้อมูลการพยากรณ์เพื่อรับข้อมูลปัจจุบันและย้อนหลังที่ครอบคลุมถึงข้อมูลที่จำเป็น
    เช่น ปริมาณน้ำฝน (rain_1h) สำหรับการสร้าง lag features.
    
    Parameters:
    - lat (float): ละติจูดของพื้นที่เป้าหมาย
    - lon (float): ลองจิจูดของพื้นที่เป้าหมาย
    - api_key (str): OpenWeatherMap API Key
    - history_hours_needed (int): จำนวนชั่วโมงย้อนหลังที่ต้องการดึงข้อมูล (ควรครอบคลุม SEQUENCE_LENGTH + PREDICTION_HORIZON_HOURS + buffer)
    
    Returns:
    - pd.DataFrame: ข้อมูลสภาพอากาศพร้อม timestamp เป็น index, หรือ DataFrame ว่างเปล่าหากเกิดข้อผิดพลาด
    """
    print(f"Fetching weather data from OpenWeatherMap for Lat: {lat}, Lon: {lon}...")
    
    # OWM forecast ให้ข้อมูลทุก 3 ชั่วโมง.
    # OWM 5-day / 3-hour forecast จะให้ข้อมูล 40 จุด (120 ชั่วโมง = 5 วัน)
    # เราจะดึงข้อมูลให้มากพอแล้วค่อยเลือกช่วงที่ต้องการ
    # ให้ดึงข้อมูลมากพอสมควรเผื่อ OWM อัปเดตช้า
    num_forecast_points = max(40, int(history_hours_needed / 3) + 5) # ดึงอย่างน้อย 40 จุด หรือตามที่ต้องการ + buffer

    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key,
        'units': 'metric', # ใช้หน่วยเมตริก (Celsius, m/s, mm)
        'cnt': num_forecast_points 
    }

    try:
        response = requests.get(config.OWM_FORECAST_URL, params=params)
        response.raise_for_status() # ตรวจสอบ HTTP errors (4xx or 5xx)
        data = response.json()

        weather_records = []
        for item in data.get('list', []):
            timestamp_utc = item['dt'] # OWM timestamps are UTC by default
            dt_object_utc = unix_to_datetime(timestamp_utc) # แปลงเป็น datetime object UTC

            weather_records.append({
                'timestamp': dt_object_utc, # เก็บเป็น UTC ก่อน
                'temperature': item['main']['temp'],
                'humidity': item['main']['humidity'],
                'pressure': item['main']['pressure'],
                'wind_speed': item['wind']['speed'],
                'wind_deg': item['wind']['deg'],
                'clouds_all': item['clouds']['all'], # เปอร์เซ็นต์เมฆคลุม
                'weather_main': item['weather'][0]['main'] if item['weather'] else None,
                'weather_description': item['weather'][0]['description'] if item['weather'] else None,
                'rain_1h': item['rain']['1h'] if 'rain' in item and '1h' in item['rain'] else 0.0, # ปริมาณฝนใน 1 ชั่วโมงที่ผ่านมา (mm)
                'rain_3h': item['rain']['3h'] if 'rain' in item and '3h' in item['rain'] else 0.0, # ปริมาณฝนใน 3 ชั่วโมงที่ผ่านมา (mm)
            })
        
        df = pd.DataFrame(weather_records).set_index('timestamp')
        df.index = df.index.tz_localize('UTC').tz_convert(config.TIMEZONE) # Localize เป็น UTC ก่อนแล้วแปลงเป็น Timezone ท้องถิ่น
        df = df.sort_index() # เรียงตามเวลา

        # OWM ให้ข้อมูลทุก 3 ชั่วโมง ดังนั้นต้อง Resample ให้เป็นรายชั่วโมงและเติมค่า
        # เราจะใช้ค่าเฉลี่ยหรือ forward-fill/interpolate เพื่อเติมช่องว่าง
        hourly_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H', tz=config.TIMEZONE)
        df_hourly = df.reindex(hourly_index)
        
        # [แก้ไข] ควรเติมค่า NaNs ให้ดีขึ้น เพราะ ffill อย่างเดียวอาจไม่เหมาะสมกับทุกคอลัมน์
        for col in ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_deg', 'clouds_all']:
            df_hourly[col] = df_hourly[col].interpolate(method='linear', limit_direction='both')
        # สำหรับฝน ควร ffill หรือใช้ 0 ถ้าไม่มีข้อมูล
        for col in ['rain_1h', 'rain_3h']:
            df_hourly[col] = df_hourly[col].ffill().fillna(0.0) # ffill ก่อนแล้วเติม 0 ถ้ายังเหลือ

        # ลบคอลัมน์ที่ไม่จำเป็นออกแต่แรก (OWM มีหลายคอลัมน์ที่ไม่ใช้ในการทำนาย)
        cols_to_keep = ['temperature', 'humidity', 'pressure', 'wind_speed', 'rain_1h'] # อาจเพิ่ม 'rain_3h' ถ้าใช้
        df_hourly = df_hourly[cols_to_keep]

        print(f"OpenWeatherMap data fetched and resampled to hourly. Shape: {df_hourly.shape}")
        return df_hourly

    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data from OpenWeatherMap: {e}")
        return pd.DataFrame()

# --- ฟังก์ชันจำลองสำหรับดึงข้อมูลระดับน้ำขึ้นน้ำลง (Tide Data) ---
def fetch_tide_data(start_time_local, end_time_local):
    """
    ดึงข้อมูลระดับน้ำขึ้นน้ำลง (Sea Level / Tide).
    """
    print(f"Fetching tide data (simulated) for {start_time_local.strftime('%Y-%m-%d %H:%M')} to {end_time_local.strftime('%Y-%m-%d %H:%M')}...")
    
    timestamps = pd.date_range(start=start_time_local, end=end_time_local, freq='H', tz=config.TIMEZONE)
    if timestamps.empty:
        return pd.DataFrame()

    # จำลองระดับน้ำขึ้นน้ำลงที่มีรูปแบบคลื่น (sinusoidal) + สุ่ม noise
    # รอบของน้ำขึ้นน้ำลงโดยทั่วไปคือประมาณ 12.42 ชั่วโมง (semidiurnal tide)
    time_in_hours = (timestamps - timestamps[0]).total_seconds() / 3600
    # Base level (ประมาณ 1.0 เมตร) + Amplitude (0.5 เมตร) * Sine wave
    dummy_tide = (np.sin(time_in_hours * (2 * np.pi / 12.42)) * 0.5) + 1.0 
    
    dummy_data = {
        'timestamp': timestamps,
        'sea_level_m': dummy_tide + np.random.uniform(-0.1, 0.1, len(timestamps)) # เพิ่ม noise
    }
    df = pd.DataFrame(dummy_data).set_index('timestamp')
    print(f"Simulated tide data generated. Shape: {df.shape}")
    return df

# --- ฟังก์ชันจำลองสำหรับดึงข้อมูลระดับน้ำในพื้นที่ (Local Water Level) ---
def fetch_local_water_level_data(area_id, start_time_local, end_time_local): # [แก้ไข] เปลี่ยนชื่อฟังก์ชันให้ชัดเจน
    """
    ดึงข้อมูลระดับน้ำในแม่น้ำ/คลอง/เซ็นเซอร์ในพื้นที่.
    """
    print(f"Fetching local water level data (simulated) for {area_id} from {start_time_local.strftime('%Y-%m-%d %H:%M')} to {end_time_local.strftime('%Y-%m-%d %H:%M')}...")
    
    timestamps = pd.date_range(start=start_time_local, end=end_time_local, freq='H', tz=config.TIMEZONE)
    if timestamps.empty:
        return pd.DataFrame()

    # จำลองระดับน้ำในคลอง/แม่น้ำที่อาจมีการขึ้นลงตามฝนหรือการระบาย
    day_of_year = timestamps.dayofyear
    seasonal_base = (np.sin(day_of_year * (2 * np.pi / 365)) * 0.2) + 0.8 # 0.6 - 1.0 เมตร
    
    dummy_water_level = seasonal_base + np.random.uniform(-0.05, 0.05, len(timestamps))
    
    # เพิ่มการจำลองน้ำท่วมเฉพาะช่วงเวลาที่เลือก (เพื่อให้โมเดลมีข้อมูล "น้ำท่วม" ให้เรียนรู้)
    # [แก้ไข] ใช้ค่าจาก config.py เพื่อให้สอดคล้องกับเกณฑ์การท่วม
    if start_time_local.month >= 5 and end_time_local.month <= 10: # เดือน พ.ค. - ต.ค. คือหน้าฝน
        # เพิ่มระดับน้ำในช่วงนี้ อาจจะเพิ่มให้เกินเกณฑ์ FLOOD_WATER_LEVEL_THRESHOLD_M
        flood_spike_potential = np.random.uniform(0.0, 0.5, len(timestamps)) # เพิ่ม potential flood
        dummy_water_level += flood_spike_potential
        
        # ทำให้บางช่วงเวลาเกินเกณฑ์น้ำท่วม
        dummy_water_level = pd.Series(dummy_water_level, index=timestamps)  # Convert to Pandas Series for mutability
        random_indices = np.random.choice(len(timestamps), size=int(len(timestamps) * 0.1), replace=False) # 10% ของจุดข้อมูล
        for idx in random_indices:
            # ทำให้ค่าบางค่าเกินค่า threshold เช่น 1.5 - 2.5 เมตร
            if dummy_water_level.iloc[idx] < config.FLOOD_WATER_LEVEL_THRESHOLD_M:
                dummy_water_level.iloc[idx] = np.random.uniform(config.FLOOD_WATER_LEVEL_THRESHOLD_M, config.FLOOD_WATER_LEVEL_THRESHOLD_M + 1.0)
        dummy_water_level = dummy_water_level.values  # Convert back to NumPy array if needed
    
    # จำกัดไม่ให้ระดับน้ำต่ำกว่า 0 (ต้องทำหลังจากการจำลองน้ำท่วมเสร็จแล้ว)
    dummy_water_level = np.maximum(dummy_water_level, 0.05)
    
    dummy_data = {
        'timestamp': timestamps,
        'local_water_level_m': dummy_water_level # [แก้ไข] เปลี่ยนชื่อเป็น local_water_level_m ให้ชัดเจน
    }
    df = pd.DataFrame(dummy_data).set_index('timestamp')
    print(f"Simulated local water level data generated. Shape: {df.shape}")
    return df


# --- ฟังก์ชันสำหรับดึงข้อมูลเชิงจุดจาก Google Earth Engine (GEE) ---
def get_gee_point_data(collection_path, band_name, point, start_date_utc, end_date_utc, reducer=ee.Reducer.mean(), scale=config.GEE_SCALE_METERS):
    """
    ดึงข้อมูล Time Series สำหรับจุดเดียวจาก GEE ImageCollection.
    """
    if not ee.data.is_initialized():
        print("GEE is not initialized. Cannot fetch satellite data.")
        return pd.DataFrame()

    print(f"Fetching GEE data from {collection_path} ({band_name}) for {start_date_utc.strftime('%Y-%m-%d %H:%M')} to {end_date_utc.strftime('%Y-%m-%d %H:%M')}...")
    
    try:
        collection = ee.ImageCollection(collection_path)\
            .filterDate(start_date_utc, end_date_utc)\
            .select(band_name)

        def get_value_for_image(image):
            date = image.date().format('YYYY-MM-dd HH:mm:ss')
            value = image.reduceRegion(
                reducer=reducer,
                geometry=point,
                scale=scale,
                tileScale=4 
            ).get(band_name)
            return ee.Feature(None, {'date': date, 'value': value})

        feature_collection = collection.map(get_value_for_image)
        
        # ดึงผลลัพธ์จาก GEE (เป็น blocking call)
        data_list = feature_collection.reduceColumns(
            ee.Reducer.toList(2), ['date', 'value']
        ).values().get(0).getInfo()
        if not data_list:
            print(f"No data found for {collection_path} in the specified period.")
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=['timestamp_str', band_name])
        df['timestamp'] = pd.to_datetime(df['timestamp_str'], utc=True) # GEE timestamps are UTC
        df = df.set_index('timestamp').drop(columns=['timestamp_str'])
        df = df[df[band_name].notna()] 
        df = df.astype({band_name: float}) 
        df.index = df.index.tz_convert(config.TIMEZONE) # แปลงเป็น Timezone ท้องถิ่น
        df = df.sort_index() 
        
        print(f"GEE data fetched. Shape: {df.shape}")
        return df

    except ee.EEException as e:
        print(f"GEE specific error fetching data from {collection_path}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"General error fetching GEE data from {collection_path}: {e}")
        return pd.DataFrame()


def fetch_satellite_rainfall(start_time_utc, end_time_utc):
    """ดึงข้อมูลปริมาณน้ำฝนรายชั่วโมงจาก GPM IMERG."""
    point = ee.Geometry.Point(config.TARGET_AREA_LON, config.TARGET_AREA_LAT)
    df_gpm = get_gee_point_data(
        # collection_path=config.GPM_IMERG_COLLECTION,
        collection_path=config.NASA_GPM_L3_IMERG_V07,
        band_name=config.GPM_RAINFALL_BAND,
        point=point,
        start_date_utc=start_time_utc,
        end_date_utc=end_time_utc,
        reducer=ee.Reducer.mean(), 
        scale=config.GEE_SCALE_METERS
    )
    if not df_gpm.empty:
        df_gpm = df_gpm.rename(columns={config.GPM_RAINFALL_BAND: 'satellite_rainfall_mm_hr'})
    return df_gpm

def fetch_satellite_water_extent(start_time_utc, end_time_utc, instrument='Sentinel-1'):
    """
    ดึงข้อมูลการสะสมของน้ำบนพื้นผิวโดยใช้ Sentinel-1 (Radar) หรือ Sentinel-2 (Optical).
    """
    point = ee.Geometry.Point(config.TARGET_AREA_LON, config.TARGET_AREA_LAT)
    
    if instrument == 'Sentinel-1':
        print("Fetching Sentinel-1 (SAR) water extent data...")
        
        collection = ee.ImageCollection(config.SENTINEL1_COLLECTION)\
            .filterDate(start_time_utc, end_time_utc)\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', config.SENTINEL1_BAND))\
            .filter(ee.Filter.eq('instrumentMode', config.SENTINEL1_FILTER_MODE))\
            .map(lambda image: image.clip(point.buffer(config.GEE_SCALE_METERS * 2))) # Clip image to area of interest
        print('collection...')
        def get_s1_value(image):
            date = image.date().format('YYYY-MM-dd HH:mm:ss')
            value = image.select(config.SENTINEL1_BAND).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=config.GEE_SCALE_METERS,
                tileScale=4
            ).get(config.SENTINEL1_BAND)
            return ee.Feature(None, {'date': date, 'value': value})
        
        try:
            feature_collection = collection.map(get_s1_value)
            print('feature_collection...')
            data_list = feature_collection.reduceColumns(
                ee.Reducer.toList(2), ['date', 'value']
            ).values().get(0).getInfo()
            print('data_list....')

            if not data_list:
                print(f"No Sentinel-1 data found for the specified period.")
                return pd.DataFrame()

            df = pd.DataFrame(data_list, columns=['timestamp_str', 'satellite_water_index_s1_vv'])
            df['timestamp'] = pd.to_datetime(df['timestamp_str'], utc=True)
            df = df.set_index('timestamp').drop(columns=['timestamp_str'])
            df = df[df['satellite_water_index_s1_vv'].notna()]
            df = df.astype({'satellite_water_index_s1_vv': float})
            df.index = df.index.tz_convert(config.TIMEZONE)
            df = df.sort_index()
            print(f"Sentinel-1 data fetched. Shape: {df.shape}")
            return df
        except ee.EEException as e:
            print(f"GEE specific error fetching Sentinel-1 data: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"General error fetching Sentinel-1 data: {e}")
            return pd.DataFrame()

    elif instrument == 'Sentinel-2':
        print("Fetching Sentinel-2 (Optical) water extent data (NDWI)...")
        # Sentinel-2 สำหรับการตรวจจับน้ำโดยใช้ NDWI (Normalized Difference Water Index)
        # NDWI = (Green - NIR) / (Green + NIR)
        # ค่า NDWI สูง (>0) บ่งบอกถึงพื้นที่น้ำ
        
        def calculate_ndwi_and_mask_clouds(image):
            # [แก้ไข] ปรับปรุงการกรองเมฆให้แข็งแกร่งขึ้น (S2 Cloud Masking Function)
            # นี่เป็นตัวอย่างง่ายๆ, สำหรับงานจริงควรใช้ ee.Algorithms.Sentinel2.addCloudShadowMask
            # หรือฟังก์ชันที่ซับซ้อนกว่า
            qa = image.select('QA60') # Cloud and cirrus band
            cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0)) # No clouds and no cirrus
            
            # Additional filter for CLOUDY_PIXEL_PERCENTAGE
            cloud_pixel_percentage = image.get('CLOUDY_PIXEL_PERCENTAGE')
            
            if ee.Number(cloud_pixel_percentage).lt(config.SENTINEL2_CLOUD_THRESHOLD).getInfo(): # ใช้ .getInfo() เพื่อประเมินค่าใน Python
                ndwi = image.normalizedDifference([config.SENTINEL2_BAND_GREEN, config.SENTINEL2_BAND_NIR]).rename('NDWI')
                return image.addBands(ndwi).updateMask(cloud_mask) # Apply cloud mask
            else:
                return image.addBands(ee.Image(0).rename('NDWI')).updateMask(ee.Image(0)) # ส่งภาพเปล่าถ้ามีเมฆมาก

        collection = ee.ImageCollection(config.SENTINEL2_COLLECTION)\
            .filterDate(start_time_utc, end_time_utc)\
            .filterBounds(point)\
            .map(calculate_ndwi_and_mask_clouds)
        
        def get_s2_ndwi_value(image):
            date = image.date().format('YYYY-MM-dd HH:mm:ss')
            ndwi_val = image.select('NDWI').reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=config.GEE_SCALE_METERS,
                tileScale=4
            ).get('NDWI')
            return ee.Feature(None, {'date': date, 'value': ndwi_val})
        
        try:
            feature_collection = collection.map(get_s2_ndwi_value)
            data_list = feature_collection.reduceColumns(
                ee.Reducer.toList(2), ['date', 'value']
            ).values().get(0).getInfo()

            if not data_list:
                print(f"No Sentinel-2 data found for the specified period.")
                return pd.DataFrame()

            df = pd.DataFrame(data_list, columns=['timestamp_str', 'satellite_water_index_s2_ndwi'])
            df['timestamp'] = pd.to_datetime(df['timestamp_str'], utc=True)
            df = df.set_index('timestamp').drop(columns=['timestamp_str'])
            df = df[df['satellite_water_index_s2_ndwi'].notna()]
            df = df.astype({'satellite_water_index_s2_ndwi': float})
            df.index = df.index.tz_convert(config.TIMEZONE)
            df = df.sort_index()
            print(f"Sentinel-2 data fetched. Shape: {df.shape}")
            return df
        except ee.EEException as e:
            print(f"GEE specific error fetching Sentinel-2 data: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"General error fetching Sentinel-2 data: {e}")
            return pd.DataFrame()
    else:
        print("Invalid instrument specified for water extent. Choose 'Sentinel-1' or 'Sentinel-2'.")
        return pd.DataFrame()

def fetch_satellite_soil_moisture(start_time_utc, end_time_utc):
    """ดึงข้อมูลความชื้นในดินจาก SMAP."""
    point = ee.Geometry.Point(config.TARGET_AREA_LON, config.TARGET_AREA_LAT)
    df_smap = get_gee_point_data(
        collection_path=config.SMAP_COLLECTION,
        band_name=config.SMAP_SOIL_MOISTURE_BAND,
        point=point,
        start_date_utc=start_time_utc,
        end_date_utc=end_time_utc,
        reducer=ee.Reducer.mean(),
        scale=config.GEE_SCALE_METERS
    )
    if not df_smap.empty:
        df_smap = df_smap.rename(columns={config.SMAP_SOIL_MOISTURE_BAND: 'satellite_soil_moisture'})
    return df_smap

def collect_historical_data():
    """
    รวบรวมข้อมูลประวัติศาสตร์จากแหล่งต่างๆ รวมถึงข้อมูลดาวเทียม.
    ฟังก์ชันนี้ควรถูกรันเมื่อคุณต้องการอัปเดตชุดข้อมูลประวัติศาสตร์สำหรับฝึกโมเดล.
    """
    print("Collecting historical data for model training...")
    
    end_date_local = get_current_local_time() # ใช้ฟังก์ชันจาก utils
    # start_date_local = end_date_local - timedelta(days=config.HISTORICAL_DAYS_TO_FETCH) # [แก้ไข] ใช้ config.HISTORICAL_DAYS_TO_FETCH
    start_date_local = end_date_local - timedelta(days=30)
    
    start_date_utc = start_date_local.astimezone(timezone('UTC'))
    end_date_utc = end_date_local.astimezone(timezone('UTC'))

    # --- 1. โหลดข้อมูลฐานจากไฟล์ CSV (คุณต้องเตรียมไฟล์นี้) ---
    if os.path.exists(config.HISTORICAL_DATA_FILE): # [แก้ไข] ใช้ HISTORICAL_DATA_FILE
        print(f"Loading base historical data from {config.HISTORICAL_DATA_FILE}")
        df_base = pd.read_csv(config.HISTORICAL_DATA_FILE, parse_dates=['timestamp']).set_index('timestamp')
        if df_base.index.tz is None:
            df_base.index = df_base.index.tz_localize(config.TIMEZONE)
        else:
            df_base.index = df_base.index.tz_convert(config.TIMEZONE)
        
        # df_base = df_base.loc[start_date_local:end_date_local].copy() # ใช้ .copy() เพื่อหลีกเลี่ยง SettingWithCopyWarning
        mask = (df_base.index >= start_date_local) & (df_base.index <= end_date_local)
        df_base = df_base.loc[mask].copy() # ใช้ .copy() เพื่อหลีกเลี่ยง SettingWithCopyWarning

    else:
        print(f"WARNING: Base historical data file not found at {config.HISTORICAL_DATA_FILE}.")
        print("Generating dummy base data for demonstration. PLEASE REPLACE WITH REAL DATA FOR ACCURATE MODELING!")
        dates = pd.date_range(start=start_date_local, end=end_date_local, freq='H', tz=config.TIMEZONE)
        dummy_data = {
            'temperature': np.random.uniform(25, 35, len(dates)),
            'humidity': np.random.uniform(60, 95, len(dates)),
            'pressure': np.random.uniform(1000, 1020, len(dates)),
            'wind_speed': np.random.uniform(0, 10, len(dates)),
            'rain_1h': np.random.choice([0, 0, 0, 0, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0], len(dates), 
                                        p=[0.7,0.1,0.05,0.05,0.02,0.02,0.02,0.02,0.01,0.01]),
            'sea_level_m': (np.sin((dates - dates[0]).total_seconds() / 3600 * (2 * np.pi / 12.42)) * 0.5) + 1.0 + np.random.uniform(-0.1, 0.1, len(dates)),
            'local_water_level_m': (np.sin((dates - dates[0]).total_seconds() / 3600 * (2 * np.pi / 24)) * 0.3) + 0.8 + np.random.uniform(-0.1, 0.1, len(dates)), # [แก้ไข] ชื่อคอลัมน์
        }
        df_base = pd.DataFrame(dummy_data, index=dates)
        df_base.index.name = 'timestamp'
        df_base['local_water_level_m'] = df_base['local_water_level_m'].apply(lambda x: max(x, 0.05)) 
        
        # [แก้ไข] สร้าง Directory หากยังไม่มี
        os.makedirs(os.path.dirname(config.HISTORICAL_DATA_FILE), exist_ok=True)
        df_base.to_csv(config.HISTORICAL_DATA_FILE) 
        print("Dummy base historical data generated and saved.")

    full_time_index = pd.date_range(start=start_date_local, end=end_date_local, freq='H', tz=config.TIMEZONE)
    df_base = df_base[~df_base.index.duplicated(keep='first')]  # Remove duplicate indices
    combined_df = df_base.reindex(full_time_index)
    
    # --- 2. ดึงข้อมูลจาก OpenWeatherMap ---
    weather_df = fetch_weather_data_from_owm(config.TARGET_AREA_LAT, config.TARGET_AREA_LON, config.OPENWEATHER_API_KEY, 
                                            history_hours_needed=(end_date_local - start_date_local).total_seconds() / 3600)
    if not weather_df.empty:
        # [แก้ไข] ควรใช้ update เพื่อเติมค่าเฉพาะที่ไม่มีใน combined_df
        combined_df.update(weather_df)
        combined_df = combined_df.join(weather_df, how='left', rsuffix='_owm') # Join ด้วย suffix เพื่อไม่ให้ทับคอลัมน์ที่มีอยู่
        # หลังจาก join แล้ว อาจจะต้องเลือกคอลัมน์ที่เหมาะสม
        for col in weather_df.columns:
            if f"{col}_owm" in combined_df.columns:
                # ถ้าคอลัมน์ต้นฉบับไม่มีค่า ให้ใช้ค่าจาก OWM
                combined_df[col] = combined_df[col].fillna(combined_df[f"{col}_owm"])
                combined_df = combined_df.drop(columns=[f"{col}_owm"])
    
    # --- 3. ดึงข้อมูลจำลองระดับน้ำขึ้นน้ำลงและระดับน้ำในพื้นที่ (ใช้สำหรับข้อมูลประวัติศาสตร์) ---
    tide_df = fetch_tide_data(start_date_local, end_date_local)
    if not tide_df.empty:
        combined_df = combined_df.join(tide_df, how='left', rsuffix='_tide')  # Add a suffix to avoid column overlap

    local_water_level_df = fetch_local_water_level_data("Bangkok", start_date_local, end_date_local) # [แก้ไข] ชื่อฟังก์ชัน
    if not local_water_level_df.empty:
        # ตรวจสอบว่ามี water_level_m อยู่ใน df_base แล้วหรือยัง
        if 'local_water_level_m' not in combined_df.columns or combined_df['local_water_level_m'].isnull().all():
            combined_df['local_water_level_m'] = local_water_level_df['local_water_level_m']
        else:
            combined_df['local_water_level_m'] = combined_df['local_water_level_m'].fillna(local_water_level_df['local_water_level_m'])


    # --- 4. ดึงข้อมูลดาวเทียมจาก GEE ---
    # GPM rainfall (รายชั่วโมง)
    df_gpm = fetch_satellite_rainfall(start_date_utc, end_date_utc)
    if not df_gpm.empty:
        # Resample GPM data to hourly and fill forward to match the main index
        combined_df = combined_df.join(df_gpm.resample('H').mean().ffill(), how='left') 
    
    # Sentinel-1 water extent (ไม่บ่อยเท่า GPM, อาจมีช่องว่างข้อมูลเยอะ)
    df_s1_water = fetch_satellite_water_extent(start_date_utc, end_date_utc, instrument='Sentinel-1')
    if not df_s1_water.empty:
        # สำหรับข้อมูลที่ sparse มากๆ เช่น Sentinel-1, การ resample และ ffill อาจทำให้เกิดค่าที่ซ้ำซ้อนนาน
        # พิจารณาการ interpolate หรือใช้ค่าที่ใกล้ที่สุด
        # [แก้ไข] อาจใช้ limit_area เพื่อไม่ให้ ffill/bfill ไกลเกินไป
        # Drop the existing column to avoid overlap or specify a suffix
        # if 'satellite_water_index_s1_vv' in combined_df.columns:
        #     combined_df = combined_df.drop(columns=['satellite_water_index_s1_vv'])
        combined_df = combined_df.join(df_s1_water.resample('H').mean().ffill(limit=24), how='left', rsuffix='_water_extent') # ffill ไม่เกิน 24 ชั่วโมง
    
    # SMAP soil moisture (รายวัน)
    df_smap_sm = fetch_satellite_soil_moisture(start_date_utc, end_date_utc)
    if not df_smap_sm.empty:
        # SMAP เป็นข้อมูลรายวัน, resample ให้เป็นรายชั่วโมงและ ffill
        combined_df = combined_df.join(df_smap_sm.resample('H').mean().ffill(limit=24), how='left') # ffill ไม่เกิน 24 ชั่วโมง
    
    # --- 5. จัดการค่าที่หายไป (NaNs) ใน DataFrame ที่รวมกันแล้ว ---
    print("Filling missing values in combined historical data...")
    for col in config.FEATURES + [config.TARGET]: # [แก้ไข] ใช้ config.FEATURES และ config.TARGET เพื่อระบุคอลัมน์ที่จำเป็น
        if col not in combined_df.columns:
            print(f"Warning: Column '{col}' not found in combined_df. Skipping NaN filling for this column.")
            continue # ข้ามถ้าคอลัมน์ไม่มีอยู่จริง

        if pd.api.types.is_numeric_dtype(combined_df[col]) and combined_df[col].isnull().any():
            # ใช้ interpolate สำหรับข้อมูลที่ควรต่อเนื่อง เช่น sensor readings, rainfall
            if 'rain' in col or 'level' in col or 'temperature' in col or 'humidity' in col or 'pressure' in col or 'wind' in col or 'satellite' in col or 'soil_moisture' in col:
                combined_df[col] = combined_df[col].interpolate(method='time', limit_direction='both', limit_area='inside') # ใช้ interpolate
                combined_df[col] = combined_df[col].ffill() # เติมส่วนที่เหลือด้วย ffill
                combined_df[col] = combined_df[col].bfill() # เติมส่วนที่เหลือด้วย bfill
                
                # ถ้ายังเหลือ NaN (เช่น ที่จุดเริ่มต้น/สิ้นสุดของ DataFrame หรือช่วงที่หายไปนานมาก)
                if combined_df[col].isnull().any():
                    if 'rain' in col or 'satellite_rainfall' in col:
                        combined_df[col] = combined_df[col].fillna(0.0) # ฝนไม่มีก็ 0
                    elif 'water_level' in col or 'sea_level' in col:
                        combined_df[col] = combined_df[col].fillna(combined_df[col].mean()) # เติมด้วยค่าเฉลี่ย
                    else:
                        combined_df[col] = combined_df[col].fillna(combined_df[col].median()) # เติมด้วยค่ามัธยฐาน
            else: # สำหรับคอลัมน์อื่นๆ ที่อาจเป็น categorical หรือไม่สำคัญมากนัก อาจใช้ 0 หรือ mode
                combined_df[col] = combined_df[col].fillna(0) 

    # ตรวจสอบอีกครั้งเพื่อความแน่ใจ
    print(f"Combined historical data shape after filling NaNs: {combined_df.shape}")
    print(f"Number of NaNs after processing: {combined_df.isnull().sum().sum()}") # ควรจะเป็น 0 หรือน้อยที่สุด

    # บันทึกข้อมูลที่รวบรวมและประมวลผลเบื้องต้นแล้ว
    # [แก้ไข] บันทึกไปที่ PROCESSED_TRAINING_DATA_FILE
    os.makedirs(os.path.dirname(config.PROCESSED_TRAINING_DATA_FILE), exist_ok=True)
    combined_df.to_csv(config.PROCESSED_TRAINING_DATA_FILE) 
    print(f"Historical data collection complete. Data saved to {config.PROCESSED_TRAINING_DATA_FILE}")
    return combined_df

def get_realtime_data_for_prediction():
    """
    ดึงข้อมูล Real-time ล่าสุดสำหรับใช้ในการพยากรณ์ ซึ่งรวมถึงข้อมูลดาวเทียม.
    ฟังก์ชันนี้จะส่งคืน DataFrame ที่มีข้อมูลดิบสำหรับช่วงเวลาที่โมเดลต้องการ (SEQUENCE_LENGTH ชั่วโมงย้อนหลัง)
    รวมถึงข้อมูลชั่วโมงปัจจุบัน
    """
    print("Fetching real-time data for prediction (including satellite data)...")
    
    current_time_local = get_current_local_time()
    # กำหนดช่วงเวลาที่ต้องการ: ข้อมูลย้อนหลัง SEQUENCE_LENGTH ชั่วโมง + ชั่วโมงปัจจุบัน
    # เพื่อให้มีข้อมูลเพียงพอสำหรับสร้าง Lag Features และ Rolling Statistics
    # [แก้ไข] ใช้ config.SEQUENCE_LENGTH และ config.PREDICTION_HORIZON_HOURS
    # เพิ่ม buffer สัก 2-3 ชั่วโมง เผื่อข้อมูล GEE/OWM อัปเดตช้า
    buffer_hours = 5 
    start_time_window_local = current_time_local - timedelta(hours=config.SEQUENCE_LENGTH + config.PREDICTION_HORIZON_HOURS + buffer_hours) 
    
    start_time_window_utc = start_time_window_local.astimezone(timezone('UTC'))
    current_time_utc = current_time_local.astimezone(timezone('UTC'))

    # --- 1. ดึงข้อมูล OpenWeatherMap ---
    weather_df = fetch_weather_data_from_owm(config.TARGET_AREA_LAT, config.TARGET_AREA_LON, config.OPENWEATHER_API_KEY, 
                                            history_hours_needed=config.SEQUENCE_LENGTH + config.PREDICTION_HORIZON_HOURS + buffer_hours) 
    
    # --- 2. ดึงข้อมูลระดับน้ำขึ้นน้ำลงและระดับน้ำในพื้นที่ (Real-time, ต้องเชื่อมต่อจริง) ---
    tide_df = fetch_tide_data(start_time_window_local, current_time_local)
    local_water_level_df = fetch_local_water_level_data("Bangkok", start_time_window_local, current_time_local) # [แก้ไข] ชื่อฟังก์ชัน
    
    # --- 3. ดึงข้อมูลดาวเทียม ---
    gpm_df = fetch_satellite_rainfall(start_time_window_utc, current_time_utc)
    s1_water_df = fetch_satellite_water_extent(start_time_window_utc, current_time_utc, instrument='Sentinel-1')
    smap_sm_df = fetch_satellite_soil_moisture(start_time_window_utc, current_time_utc)

    # --- 4. รวมข้อมูลทั้งหมดเข้าด้วยกัน ---
    full_window_times = pd.date_range(start=start_time_window_local, end=current_time_local, freq='H', tz=config.TIMEZONE)
    combined_realtime_df = pd.DataFrame(index=full_window_times)

    if not weather_df.empty:
        combined_realtime_df = combined_realtime_df.join(weather_df, how='left')
    if not gpm_df.empty:
        combined_realtime_df = combined_realtime_df.join(gpm_df, how='left')
    if not s1_water_df.empty:
        combined_realtime_df = combined_realtime_df.join(s1_water_df.resample('H').mean().ffill(limit=config.SEQUENCE_LENGTH), how='left') # [แก้ไข] ffill limit
    if not smap_sm_df.empty:
        combined_realtime_df = combined_realtime_df.join(smap_sm_df.resample('H').mean().ffill(limit=config.SEQUENCE_LENGTH), how='left') # [แก้ไข] ffill limit
    if not tide_df.empty:
        combined_realtime_df = combined_realtime_df.join(tide_df, how='left')
    if not local_water_level_df.empty:
        combined_realtime_df = combined_realtime_df.join(local_water_level_df, how='left')

    # --- 5. จัดการค่าที่หายไป (NaNs) ในข้อมูล Real-time ---
    # สำหรับข้อมูล Real-time การเติมค่า NaNs สำคัญมาก
    # ใช้ ffill เพื่อใช้ค่าล่าสุดที่มีอยู่
    print("Filling missing values in real-time data...")
    for col in config.FEATURES: # [แก้ไข] ใช้ config.FEATURES เพื่อระบุคอลัมน์ที่จำเป็น
        if col not in combined_realtime_df.columns:
            print(f"Warning: Column '{col}' not found in real-time combined_df. This might affect prediction accuracy.")
            # ถ้าคอลัมน์สำคัญหายไป ควรเติมด้วยค่า default หรือ 0 เพื่อไม่ให้เกิด error ใน preprocessing
            # ในสถานการณ์จริง คุณอาจต้อง Raise Error หรือแจ้งเตือนให้ชัดเจนขึ้น
            combined_realtime_df[col] = np.nan # เพิ่มคอลัมน์ที่หายไปก่อนเติม
            
        if pd.api.types.is_numeric_dtype(combined_realtime_df[col]) and combined_realtime_df[col].isnull().any():
            # พยายาม interpolate ก่อน แล้วค่อย ffill/bfill
            combined_realtime_df[col] = combined_realtime_df[col].interpolate(method='time', limit_direction='both', limit_area='inside')
            combined_realtime_df[col] = combined_realtime_df[col].ffill() 
            combined_realtime_df[col] = combined_realtime_df[col].bfill() 
            
            # ถ้ายังเหลือ NaN (ที่จุดเริ่มต้นของ DataFrame หรือช่องว่างขนาดใหญ่มาก)
            if combined_realtime_df[col].isnull().any():
                if 'rain' in col or 'satellite_rainfall' in col:
                    combined_realtime_df[col] = combined_realtime_df[col].fillna(0.0)
                elif 'water_level' in col or 'sea_level' in col:
                    combined_realtime_df[col] = combined_realtime_df[col].fillna(combined_realtime_df[col].mean()) # อาจใช้ค่าเฉลี่ยจากข้อมูลย้อนหลัง (ถ้ามี)
                else:
                    combined_realtime_df[col] = combined_realtime_df[col].fillna(combined_realtime_df[col].median()) # หรือค่ามัธยฐาน

    # [แก้ไข] กรองเฉพาะข้อมูลที่อยู่ในช่วงที่ต้องการจริงๆ (เผื่อไว้ถ้าดึงข้อมูลมาเกิน)
    # เนื่องจากเราต้องการข้อมูลดิบที่ครบถ้วนสำหรับ SEQUENCE_LENGTH + 1 จุดสุดท้าย
    # เพื่อให้ data_preprocessing สามารถสร้าง lagged features ได้
    combined_realtime_df = combined_realtime_df.loc[start_time_window_local:current_time_local].copy()

    # ตรวจสอบว่าข้อมูลมีจำนวนชั่วโมงเพียงพอสำหรับ SEQUENCE_LENGTH
    if len(combined_realtime_df) < config.SEQUENCE_LENGTH + 1: # [แก้ไข] +1 เพื่อให้ได้จุดปัจจุบันด้วย
        print(f"Warning: Not enough data points ({len(combined_realtime_df)}) for required SEQUENCE_LENGTH ({config.SEQUENCE_LENGTH}). This may cause issues in preprocessing.")
        # อาจจะต้องพิจารณาว่าจะ raise error หรือเติมค่าด้วยวิธีอื่นหากข้อมูลไม่พอจริงๆ

    print(f"Real-time data collection complete. Final shape: {combined_realtime_df.shape}")
    return combined_realtime_df
