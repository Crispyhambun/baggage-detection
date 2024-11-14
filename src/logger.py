import logging
import os
from datetime import datetime
import sys

# Get the log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create 'Logs' directory if it doesn't exist
logs_path = os.path.join(os.getcwd(), "Logs")
os.makedirs(logs_path, exist_ok=True)

# Define the complete file path for the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="(%(asctime)s) - %(lineno)s  %(name)s - %(levelname)s - %(message)s ",
    level=logging.INFO
)

# Example logging usage
logging.info("Logger initialized successfully.")
