import logging
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)

# Define custom log colors
LOG_COLORS = {
    'DEBUG': Fore.CYAN,
    'INFO': Fore.GREEN,
    'WARNING': Fore.YELLOW,
    'ERROR': Fore.RED,
    'CRITICAL': Back.RED + Fore.WHITE
}

class ColoredFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format(self, record):
        log_color = LOG_COLORS.get(record.levelname, '')
        log_message = super().format(record)
        return log_color + log_message

# Set up the logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)

# formatter = ColoredFormatter('%(levelname)s: %(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)

# # Test the logger
# logger.debug("This is a debug message")
# logger.info("This is an info message")
# logger.warning("This is a warning message")
# logger.error("This is an error message")
# logger.critical("This is a critical message")
