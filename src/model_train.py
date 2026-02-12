import logging

from ftta.cli import run_model_train_cli

LOG_LEVEL = logging.DEBUG

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')

if __name__ == "__main__":
    run_model_train_cli()


