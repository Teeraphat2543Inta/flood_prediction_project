# gee_connection.py
import os
import json
from typing import Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import ee

from config import config

class GEEConnection:
    def __init__(self, json_key_path: Optional[str] = None):
        self._ee = None
        """
        Initialize GEE connection with service account credentials
        Args:
            json_key_path: Path to service account JSON key file
        """
        try:
            # Get JSON key path from env or parameter
            self.json_key_path = json_key_path or config.GEE_JSON_KEY
            
            if not self.json_key_path:
                raise ValueError("JSON key path not provided")
            
            # Load and validate JSON credentials
            credentials = self._load_credentials()
            
            # Initialize Earth Engine with service account
            credentials = ee.ServiceAccountCredentials(
                credentials['client_email'],
                key_data=json.dumps(credentials)
            )
            ee.Initialize(credentials)
            self._ee = ee
            # Test connection
            print(ee.String('Successfully connected to Earth Engine!').getInfo())
            
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Earth Engine: {str(e)}")

    def _load_credentials(self) -> dict:
        """Load and validate JSON credentials file"""
        try:
            with open(self.json_key_path) as f:
                credentials = json.load(f)
                
            required_keys = ['client_email', 'private_key', 'project_id']
            if not all(key in credentials for key in required_keys):
                raise ValueError("Invalid credentials format")
                
            return credentials
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading credentials file: {str(e)}")
    
    @property
    def ee(self):
        """Return initialized ee module"""
        if self._ee is None:
            raise RuntimeError("Earth Engine not initialized")
        return self._ee