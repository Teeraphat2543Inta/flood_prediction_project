import pandas as pd
import os
import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import config
from src.data_ingestion import fetch_satellite_rainfall

START_DATE = datetime.datetime(2023, 6, 21)
END_DATE = datetime.datetime(2023, 6, 25)

historical_df = fetch_satellite_rainfall(START_DATE, END_DATE)

print(historical_df.head())