import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import datetime

from config import config

from src.data_ingestion import fetch_satellite_rainfall
# from src.utils import get_current_local_time


start_date_str = "2023-06-21 06:51:36.297017+00:00"
end_date_str = "2025-06-20 06:51:36.297017+00:00"

start_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S.%f%z")
end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S.%f%z")

start_date = datetime.datetime(2023, 6, 21)
end_date = datetime.datetime(2023, 6, 25)
historical_df = fetch_satellite_rainfall(start_date, end_date)

print(historical_df.head())
# print(get_current_local_time())