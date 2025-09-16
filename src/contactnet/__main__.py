from src.util import log_exceptions
import logging
logger = logging.getLogger(__file__)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ContactNet Main Entry Point")
    # data-gen
    parser.add_argument("--datagen", action="store_true", help="Generate data")
    args = parser.parse_known_args()
    used_args, unknown_args = args

    with log_exceptions(logger):
        if used_args.datagen:
            from src.contactnet import datagen
            datagen.main()
        
        else:
            raise ValueError("No valid arguments provided. Use --help for more information.")