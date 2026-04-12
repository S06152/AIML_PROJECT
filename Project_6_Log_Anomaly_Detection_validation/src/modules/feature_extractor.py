import sys
import pandas as pd
from src.utils.logger import logging
from src.utils.exception import CustomException

class FeatureExtractor:
    """
    FeatureExtractor class transforms parsed CAN log dataframe
    into numerical machine learning features used for anomaly detection.

    The extracted features help detect anomalies related to:
    - Abnormal timing
    - Unexpected CAN bus channel
    - CAN ID spoofing
    - Tx/Rx mismatch
    - Wrong DLC
    - Payload manipulation
    - Message flooding
    - Wrong sequence of CAN IDs
    
    | Parameter    | Possible Anomaly    | Feature Used                     | Description                                                 |
    | ------------ | ------------------- | -------------------------------- | ----------------------------------------------------------- |
    | Timestamp    | Abnormal timing     | `time_diff`, `abnormal_timing`   | Time when the message occurred.                             |
    | Channel      | Unexpected bus      | `channel`                        | Indicates which CAN bus channel transmitted the message     |
    | CAN ID       | Unknown / spoofed   | `can_id_int`, `id_frequency`     | CanID of the message                                        |
    | Direction    | Tx/Rx mismatch      | `direction_flag`                 | Indicates Transmit or Receive.                              |
    | DLC          | Wrong data length   | `dlc`                            | Length of DataFrame                                         |
    | Payload      | Signal manipulation | `payload_sum`, `payload_entropy` | Actual signal Data                                          |
    | Frequency    | Flooding            | `id_frequency`                   | Interval of can message like ID 520 → every 10ms            |
    | Sequence     | Wrong order         | `sequence_change`                | Certain messages follow a sequence like ID 100→ID 200→ID 300|
    """

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract machine learning features from parsed CAN log data.

        Parameters
        ----------
        df : pd.DataFrame
            Parsed CAN log dataframe.

        Returns
        -------
        pd.DataFrame
            Feature matrix for ML model.
        """

        try:
            logging.info("Starting feature extraction")

            df = df.copy()

            # -------------------------------------------------
            # 1. Convert CAN ID from hexadecimal → integer
            # Example: '3D8' → 984
            # -------------------------------------------------
            df["can_id_int"] = df["can_id"].astype(str).apply(lambda x: int(x, 16))
            logging.info("CAN ID converted from hex to integer")

            # -------------------------------------------------
            # 2. Payload Sum Feature
            # Example: "01 02 03 04" -> 10
            # Helps detect signal manipulation
            # -------------------------------------------------
            df["payload_sum"] = df["data"].apply(
                lambda x: sum(int(byte, 16) for byte in str(x).split())
            )

            logging.info("Payload sum feature generated")

            # -------------------------------------------------
            # 3. Timestamp Difference
            # Detect abnormal message timing
            # -------------------------------------------------
            df["time_diff"] = df["timestamp"].diff().fillna(0)

            logging.info("Time difference feature calculated")

            # -------------------------------------------------
            # 4. Abnormal Timing Detection
            # Messages occurring too quickly or too slowly
            # -------------------------------------------------
            avg_time = df["time_diff"].mean()

            df["abnormal_timing"] = df["time_diff"].apply(
                lambda x: 1 if x > (avg_time * 5) else 0
            )

            logging.info("Abnormal timing feature created")

            # -------------------------------------------------
            # 5. Channel Feature
            # Detect unexpected CAN bus
            # -------------------------------------------------
            if "channel" in df.columns:
                df["channel"] = df["channel"]
            else:
                df["channel"] = 0

            logging.info("Channel feature processed")

            # -------------------------------------------------
            # 6. Direction Encoding
            # Tx → 1 , Rx → 0
            # -------------------------------------------------
            if "direction" in df.columns:
                df["direction_flag"] = df["direction"].apply(
                    lambda x: 1 if str(x).lower() == "tx" else 0
                )
            else:
                df["direction_flag"] = 0

            logging.info("Direction encoded")

            # -------------------------------------------------
            # 7. CAN ID Frequency
            # Detect flooding attacks
            # -------------------------------------------------
            id_counts = df["can_id_int"].value_counts()

            df["id_frequency"] = df["can_id_int"].map(id_counts)

            logging.info("CAN ID frequency feature created")

            # -------------------------------------------------
            # 8. Sequence Change Detection
            # Detect abnormal message ordering
            # -------------------------------------------------
            df["prev_id"] = df["can_id_int"].shift(1)

            df["sequence_change"] = (
                df["can_id_int"] != df["prev_id"]
            ).astype(int)

            df["sequence_change"] = df["sequence_change"].fillna(0)

            logging.info("Sequence change feature generated")

            # -------------------------------------------------
            # 9. Select features for ML model
            # -------------------------------------------------
            features = df[
                [
                    "can_id_int",
                    "channel",
                    "direction_flag",
                    "dlc",
                    "payload_sum",
                    "time_diff",
                    "abnormal_timing",
                    "id_frequency",
                    "sequence_change",
                ]
            ]

            logging.info(f"Feature extraction completed. Total features: {features.shape[1]}")

            return features

        except Exception as e:
            logging.error("Error occurred during feature extraction")
            raise CustomException(e, sys)