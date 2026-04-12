"""
exception.py — Custom exception class for AUTOSAR MAS project.

Captures the original traceback details (file, line, message)
and re-raises them in a clean, structured format for logging.
"""

import sys
from src.utils.logger import logger


class CustomException(Exception):
    """
    Custom exception that enriches the standard Python exception with
    file name and line number information extracted from the traceback.

    Usage:
        try:
            risky_operation()
        except Exception as e:
            raise CustomException(e, sys) from e
    """

    def __init__(self, error: Exception, error_detail: sys) -> None:
        """
        Initialize CustomException with enriched error message.

        Args:
            error        (Exception): The original caught exception.
            error_detail (sys)      : The sys module (provides exc_info).
        """
        super().__init__(str(error))

        # Extract traceback details
        _, _, exc_tb = error_detail.exc_info()

        if exc_tb is not None:
            self._file_name: str = exc_tb.tb_frame.f_code.co_filename
            self._line_number: int = exc_tb.tb_lineno
        else:
            self._file_name = "unknown"
            self._line_number = -1

        self._error_message: str = (
            f"Error in [{self._file_name}] "
            f"at line [{self._line_number}]: "
            f"{str(error)}"
        )

        logger.error(self._error_message)

    def __str__(self) -> str:
        return self._error_message
