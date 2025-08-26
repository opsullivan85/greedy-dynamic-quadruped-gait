from src.util import log_exceptions
from src.gaitnet import logger

@log_exceptions(logger)
def main():
    print("Hello, GaitNet!")

if __name__ == "__main__":
    main()