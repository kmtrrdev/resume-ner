import logging
import sys

def setup_logging():
    """Configure structured logging for the backend"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Optional: Setup watchtower for CloudWatch integration here
    # import watchtower
    # logging.getLogger().addHandler(watchtower.CloudWatchLogHandler())
    
    logger = logging.getLogger("backend")
    logger.info("Structured logging configured.")
    return logger
