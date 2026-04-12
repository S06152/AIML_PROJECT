import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.main import Pipeline

if __name__ == "__main__":
    """
    Entry point of the CAN Log Anomaly Detection application.

    This script performs the following operations:

    1. Initialize the Pipeline
    2. Train the anomaly detection model using normal CAN logs
    3. Detect anomalies in defect/attack CAN logs
    4. Save detected anomalies to CSV
    """

    try:
        # ---------------------------------------------------
        # Application start log
        # ---------------------------------------------------
        logging.info("Starting CAN Log Anomaly Detection Application")

        # ---------------------------------------------------
        # Step 1: Initialize Pipeline
        # ---------------------------------------------------
        logging.info("Initializing pipeline")

        pipeline = Pipeline()

        logging.info("Pipeline initialized successfully")

        # ---------------------------------------------------
        # Step 2: Train model using NORMAL logs
        # ---------------------------------------------------
        logging.info("Starting model training using normal logs")

        pipeline.train()

        logging.info("Model training completed")

        # ---------------------------------------------------
        # Step 3: Detect anomalies using DEFECT logs
        # ---------------------------------------------------
        logging.info("Starting anomaly detection using defect logs")

        pipeline.anomaly()

        logging.info("Anomaly detection completed successfully")

        # ---------------------------------------------------
        # Application finished successfully
        # ---------------------------------------------------
        logging.info("CAN Log Anomaly Detection Application finished successfully")

    except Exception as e:
        logging.error("Application execution failed")
        raise CustomException(e, sys)