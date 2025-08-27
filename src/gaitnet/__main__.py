from src.util import log_exceptions
import logging
logger = logging.getLogger(__file__)

@log_exceptions(logger)
def main():
    print("Hello, GaitNet!")

if __name__ == "__main__":
    main()