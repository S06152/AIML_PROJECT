import re
import sys
import pandas as pd
from src.utils.logger import logging
from src.utils.exception import CustomException

class LogParser:
    """
    LogParser is responsible for parsing Vector CANoe .asc log files
    and extracting only valid CAN frame records.
    """

    def __init__(self):
        """
        Initialize the CAN frame regex pattern.
        This pattern identifies valid CAN frame lines in the ASC file.
        """

        try:
            # Regex pattern explanation:
            # ^\s*                -> Ignore leading spaces
            # (\d+\.\d+)          -> Timestamp (example: 1.030864)
            # (\d+)               -> Channel number (example: 1)
            # ([0-9A-Fa-f]+)      -> CAN ID (example: A8, 3D8)
            # (Tx|Rx)             -> Direction (Transmit / Receive)
            # d                   -> Data frame indicator
            # (\d+)               -> DLC (Data Length Code)
            # ((?:[0-9A-Fa-f]{2}\s+){0,8}) -> Data bytes (0–8 bytes)

            self.can_pattern = re.compile(
                r"^\s*(\d+\.\d+)\s+(\d+)\s+([0-9A-Fa-f]+)\s+(Tx|Rx)\s+d\s+(\d+)\s+((?:[0-9A-Fa-f]{2}\s+){0,8})"
            )

            logging.info("LogParser initialized successfully.")

        except Exception as e:
            logging.error("Failed to initialize LogParser.")
            raise CustomException(e, sys)

    def parse(self, lines):
        """
        Parse raw ASC log lines and extract CAN frame information.

        Parameters
        ----------
        lines : list
            List containing raw lines read from the ASC log file.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing structured CAN frame data.
        """

        try:
            logging.info("Starting CAN log parsing.")

            records = []

            for line_number, line in enumerate(lines):

                # Remove leading and trailing spaces/newline characters
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Try matching the CAN frame regex pattern
                match = self.can_pattern.match(line)

                # If line does not match CAN frame format → skip it
                if not match:
                    logging.debug(f"Skipping non-CAN line {line_number}: {line}")
                    continue

                try:
                    # Extract CAN frame fields from regex groups
                    timestamp = float(match.group(1))
                    channel = int(match.group(2))
                    can_id = match.group(3)
                    direction = match.group(4)
                    dlc = int(match.group(5))
                    data = match.group(6).strip()

                    # Store parsed record
                    records.append(
                        {
                            "timestamp": timestamp,
                            "channel": channel,
                            "can_id": can_id,
                            "direction": direction,
                            "dlc": dlc,
                            "data": data,
                        }
                    )

                except Exception as parse_error:
                    logging.warning(
                        f"Failed to parse line {line_number}: {line} | Error: {parse_error}"
                    )

            # Convert extracted records to pandas DataFrame
            df = pd.DataFrame(records)
            print(df.head(5))

            logging.info(f"Total CAN frames extracted: {len(df)}")

            return df

        except Exception as e:
            logging.error("Error occurred while parsing CAN log file.")
            raise CustomException(e, sys)