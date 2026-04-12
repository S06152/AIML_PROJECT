import sys
import pandas as pd
from src.utils.logger import logging
from src.utils.exception import CustomException
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class AnomalyDetector:
    """
    Responsible for detecting anomalies in CAN log data
    using a trained Isolation Forest model.
    """

    def anomaly_detect(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        model: IsolationForest,
        scaler: StandardScaler,
    ):
        """
        Detect anomalies using the trained model.

        Parameters
        ----------
        df : pd.DataFrame
            Original dataframe containing parsed CAN log data.

        features : pd.DataFrame
            Feature dataset used by the model for prediction.

        model : IsolationForest
            Trained Isolation Forest model.

        scaler : StandardScaler
            Fitted scaler used during model training.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only detected anomalies.
        """

        try:
            logging.info("Starting anomaly detection process")

            # ---------------------------------------------------
            # Validate inputs
            # ---------------------------------------------------
            if df.empty or features.empty:
                raise ValueError("Input dataframe or feature set is empty")

            logging.info("Input data validation completed")

            # ---------------------------------------------------
            # Scale features using trained scaler
            # ---------------------------------------------------
            logging.info("Scaling features for prediction")

            scaled_features = scaler.transform(features)

            # ---------------------------------------------------
            # Run anomaly prediction
            # ---------------------------------------------------
            logging.info("Running model prediction")

            preds = model.predict(scaled_features)

            logging.info("Model prediction completed")

            # ---------------------------------------------------
            # Add prediction results to dataframe
            # 1  -> Normal data
            # -1 -> Anomaly
            # ---------------------------------------------------
            df["anomaly"] = preds

            logging.info("Anomaly column added to dataframe")

            # ---------------------------------------------------
            # Filter anomaly rows
            # ---------------------------------------------------
            anomalies = df[df["anomaly"] == -1]

            logging.info(f"Total anomalies detected: {len(anomalies)}")

            return anomalies

        except Exception as e:
            logging.error("Error occurred during anomaly detection")
            raise CustomException(e, sys)