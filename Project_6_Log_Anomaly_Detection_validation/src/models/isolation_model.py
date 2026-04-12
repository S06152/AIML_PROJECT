import os
import joblib
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class IsolationModel:
    """
    Manages the lifecycle of the Isolation Forest model.

    Responsibilities
    ----------------
    - Initialize model
    - Train model
    - Save / load model
    - Save / load scaler
    - Predict anomalies
    """

    def __init__(self, contamination: float):
        """
        Initialize Isolation Forest model.

        Parameters
        ----------
        contamination : float
            Expected proportion of anomalies in dataset.
        """

        try:
            logging.info("Initializing Isolation Forest model")

            # Initialize Isolation Forest
            self.model = IsolationForest(
                contamination=contamination,
                n_estimators=200,
                max_samples="auto",
                random_state=42
            )

            # Feature normalization
            self._scaler = StandardScaler()

            logging.info("Isolation Forest model initialized successfully")

        except Exception as e:
            logging.error("Error initializing Isolation Forest model")
            raise CustomException(e, sys)

    def train(self, features):
        """
        Train the Isolation Forest model.

        Parameters
        ----------
        features : DataFrame
            Feature matrix used for training
        """

        try:
            logging.info("Scaling training features")

            scaled_features = self._scaler.fit_transform(features)

            logging.info("Starting model training")

            self.model.fit(scaled_features)

            logging.info("Model training completed successfully")

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)

    def save_model(self, path):
        """
        Save trained model to disk.
        """

        try:
            logging.info(f"Saving model to: {path}")

            joblib.dump(self.model, path)

            logging.info("Model saved successfully")

        except Exception as e:
            logging.error("Error occurred while saving model")
            raise CustomException(e, sys)

    def load_model(self, path):
        """
        Load trained model from disk.
        """

        try:
            logging.info(f"Loading model from: {path}")

            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")

            self.model = joblib.load(path)

            logging.info("Model loaded successfully")

        except Exception as e:
            logging.error("Error occurred while loading model")
            raise CustomException(e, sys)

    def save_scaler(self, path):
        """
        Save StandardScaler to disk.
        """

        try:
            logging.info(f"Saving scaler to: {path}")

            joblib.dump(self._scaler, path)

            logging.info("Scaler saved successfully")

        except Exception as e:
            logging.error("Error occurred while saving scaler")
            raise CustomException(e, sys)

    def load_scaler(self, path):
        """
        Load StandardScaler from disk.
        """

        try:
            logging.info(f"Loading scaler from: {path}")

            if not os.path.exists(path):
                raise FileNotFoundError(f"Scaler file not found: {path}")

            self._scaler = joblib.load(path)

            logging.info("Scaler loaded successfully")

        except Exception as e:
            logging.error("Error occurred while loading scaler")
            raise CustomException(e, sys)