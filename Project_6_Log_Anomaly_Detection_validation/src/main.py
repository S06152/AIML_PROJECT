import sys
from src.config.config import Config
from src.modules.log_reader import LogReader
from src.modules.log_parser import LogParser
from src.modules.feature_extractor import FeatureExtractor
from src.modules.anomaly_detector import AnomalyDetector
from src.models.isolation_model import IsolationModel
from src.utils.file_utils import FileUtils
from src.utils.logger import logging
from src.utils.exception import CustomException

class Pipeline:
    """
    Pipeline orchestrates the complete CAN log anomaly detection workflow.

    Workflow Steps
    --------------
    1. Read CAN log file (.asc)
    2. Parse raw logs into structured dataframe
    3. Extract ML features
    4. Train anomaly detection model
    5. Detect anomalies
    6. Save anomaly results
    """

    def __init__(self):
        """Initialize pipeline modules."""

        try:
            logging.info("Initializing Anomaly Detection Pipeline")

            # ------------------------------------------
            # Initialize pipeline components
            # ------------------------------------------

            # Reads raw CAN log file
            self.reader = LogReader()

            # Parses raw log lines into structured data
            self.parser = LogParser()

            # Extracts ML features from parsed data
            self.extractor = FeatureExtractor()

            # Detects anomalies using trained model
            self.detector = AnomalyDetector()

            # Isolation Forest anomaly detection model
            self.model = IsolationModel(Config.MODEL_CONTAMINATION)

            logging.info("Pipeline initialized successfully")

        except Exception as e:
            logging.error("Error occurred during pipeline initialization")
            raise CustomException(e, sys)

    # -----------------------------------------------------
    # TRAINING PIPELINE
    # -----------------------------------------------------
    def train(self):
        """Train anomaly detection model using normal CAN logs."""

        try:
            logging.info("===== TRAINING PIPELINE STARTED =====")

            # Step 1: Read normal CAN log
            logging.info("Step 1: Reading normal CAN log file")

            lines = self.reader.read_logs(Config.NORMAL_LOG_PATH)

            # Step 2: Parse logs
            logging.info("Step 2: Parsing CAN log data")

            df = self.parser.parse(lines)

            logging.info(f"Total CAN frames parsed: {len(df)}")

            # Step 3: Feature extraction
            logging.info("Step 3: Extracting ML features")

            features = self.extractor.extract(df)

            logging.info(f"Feature matrix shape: {features.shape}")

            # Step 4: Train model
            logging.info("Step 4: Training Isolation Forest model")

            self.model.train(features)

            # Step 5: Save model and scaler
            logging.info("Step 5: Saving trained model")

            self.model.save_model(Config.MODEL_PATH)
            self.model.save_scaler(Config.SCALER_PATH)

            logging.info("Training pipeline completed successfully")

        except Exception as e:
            logging.error("Error occurred during training pipeline")
            raise CustomException(e, sys)

    # -----------------------------------------------------
    # ANOMALY DETECTION PIPELINE
    # -----------------------------------------------------
    def anomaly(self):
        """Detect anomalies using defect CAN logs."""

        try:
            logging.info("===== ANOMALY DETECTION STARTED =====")

            # Step 1: Load trained model
            logging.info("Loading trained model and scaler")

            self.model.load_model(Config.MODEL_PATH)
            self.model.load_scaler(Config.SCALER_PATH)

            # Step 2: Read defect log
            logging.info("Step 2: Reading defect CAN log")

            lines = self.reader.read_logs(Config.DEFECT_LOG_PATH)

            # Step 3: Parse log
            logging.info("Step 3: Parsing defect CAN frames")

            df = self.parser.parse(lines)

            logging.info(f"Total CAN frames parsed: {len(df)}")

            # Step 4: Feature extraction
            logging.info("Step 4: Extracting ML features")

            features = self.extractor.extract(df)

            logging.info(f"Feature matrix shape: {features.shape}")

            # Step 5: Run anomaly detection
            logging.info("Step 5: Running anomaly detection")

            anomalies = self.detector.anomaly_detect(
                df,
                features,
                self.model.model,
                self.model._scaler
            )

            logging.info(f"Total anomalies detected: {len(anomalies)}")

            # Step 6: Save results
            logging.info("Step 6: Saving anomaly results")

            FileUtils.save_csv(anomalies, Config.OUTPUT_FILE)

            logging.info("Anomaly detection pipeline completed successfully")

        except Exception as e:
            logging.error("Error occurred during anomaly detection pipeline")
            raise CustomException(e, sys)