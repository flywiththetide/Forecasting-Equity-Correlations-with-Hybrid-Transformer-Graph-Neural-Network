
import logging
import os

def setup_logger(save_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(save_path, 'run.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('THGNN')
