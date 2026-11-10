
from chatbot.logger_config import setup_logger


logger = setup_logger()

logger.info("FAQ chatbot started.")
logger.warning("Example warning: no relevant FAQ found.")
logger.error("Example error: simulated generation failure.")

print("Logging test completed.")
