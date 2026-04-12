import os
import sys
import pandas as pd
from src.utils.logger import logging
from src.utils.exception import CustomException

class FileUtils:
    """
    FileUtils class provides utility functions
    for handling file operations such as saving output files.
    """

    @staticmethod
    def save_csv(df: pd.DataFrame, path: str) -> None:
        """
        Save a pandas DataFrame to a CSV file.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that needs to be saved.

        path : str
            Output file path where CSV will be stored.
        """

        try:
            logging.info("Starting CSV file save operation")

            # ---------------------------------------------------
            # Create directory if it does not exist
            # Example: output/anomalies.csv -> creates 'output/'
            # ---------------------------------------------------
            directory = os.path.dirname(path)

            if directory:
                os.makedirs(directory, exist_ok=True)
                logging.info(f"Directory ensured: {directory}")

            # ---------------------------------------------------
            # Save dataframe to CSV file
            # ---------------------------------------------------
            df.to_csv(path, index=False)

            logging.info(f"CSV file successfully saved at: {path}")

        except Exception as e:
            logging.error("Error occurred while saving CSV file")
            raise CustomException(e, sys)