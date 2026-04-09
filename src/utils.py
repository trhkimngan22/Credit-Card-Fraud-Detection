import yaml
import logging
import os

# Define the root directory dynamically based on structure (one level up from src/)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_config(config_file="config.yaml"):
    """Load the project configuration from YAML file."""
    config_path = os.path.join(ROOT_DIR, config_file)
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def setup_logger(name):
    """
    Setup a custom logger that writes to both Console and a log file.
    The log file path is defined in config.yaml.
    """
    config = load_config()
    log_path = os.path.join(ROOT_DIR, config['paths']['log_file'])
    
    # Create directory for logs if doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Ensure handlers are not duplicated upon multiple imports
    if not logger.hasHandlers():
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # File handler
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger
