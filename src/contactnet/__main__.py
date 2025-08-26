from src.util import log_exceptions
from src import logger

@log_exceptions(logger)
def main():
    print("Hello, ContactNet!")

if __name__ == "__main__":
    main()