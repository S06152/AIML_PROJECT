import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.main import LangGraphApp

if __name__=="__main__":
    """
    Wrapper function to run the application.
    """
    try:
        app = LangGraphApp()
        app.run()

    except Exception as e:
        logging.error(f"Critical error in app execution: {str(e)}")
        raise CustomException(e, sys)