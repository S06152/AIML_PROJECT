import os
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException

class Config:
    """
    Configuration class for the CAN Log Anomaly Detection project.
    Stores file paths and ML model parameters in one place.
    """

    try:
        # ---------------------------------------------------
        # Project Root Directory
        # ---------------------------------------------------
        # Dynamically determine the base directory of the project
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logging.info(f"Project base directory detected: {BASE_DIR}")

        # ---------------------------------------------------
        # Input CAN Log Files
        # ---------------------------------------------------

        # Path to normal CAN log file used for training
        NORMAL_LOG_PATH = os.path.join(BASE_DIR, "data", "normal_log.asc")
        logging.info(f"Normal log path set: {NORMAL_LOG_PATH}")

        # Path to defect/attack CAN log file used for anomaly detection
        DEFECT_LOG_PATH = os.path.join(BASE_DIR, "data", "defect_log.asc")
        logging.info(f"Defect log path set: {DEFECT_LOG_PATH}")

        # ---------------------------------------------------
        # Output File
        # ---------------------------------------------------

        # File where detected anomalies will be saved
        OUTPUT_FILE = os.path.join(BASE_DIR, "output", "anomalies.csv")
        logging.info(f"Output file path set: {OUTPUT_FILE}")

        # ---------------------------------------------------
        # Model Storage Paths
        # ---------------------------------------------------

        # Path to save/load trained Isolation Forest model
        MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_forest.pkl")
        logging.info(f"Model path set: {MODEL_PATH}")

        # Path to save/load the StandardScaler
        SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
        logging.info(f"Scaler path set: {SCALER_PATH}")

        # ---------------------------------------------------
        # ML Model Parameters
        # ---------------------------------------------------

        # Expected proportion of anomalies in dataset (1%)
        MODEL_CONTAMINATION = 0.01
        logging.info(f"Model contamination set to: {MODEL_CONTAMINATION}")

        # Log successful configuration setup
        logging.info("Configuration parameters initialized successfully")

    except Exception as e:
        logging.error("Error occurred while initializing configuration parameters")
        raise CustomException(e, sys)