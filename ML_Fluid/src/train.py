import argparse
import logging
import os
from ESN import EchoStateNetwork
from config import * 
import util


def train_ESN( training_data_filename, trained_model_params, trained_model_filename ): 
    """
    Initialize and train echo state network on training data with specific training params
    and save ESN object to file
    """
    input_data_fullpath = os.path.join( PREPROCESS_INPUT_PATH_DIR, training_data_filename)
    flattened_input_data = util.flatten_time_series( util.load_data( input_data_fullpath ) ).transpose()
    esn = EchoStateNetwork(loaddata = flattened_input_data, **trained_model_params)
    esn.train()
    util.save_model( esn, os.path.join( DATA_PATH_DIR, trained_model_filename) )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', choices=CONFIGS.keys(), help='Keys from CONFIGS dict in config.py ')
    parser.add_argument(
        '-d', '--debug',
        help="Print lots of debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        '-v', '--verbose',
        help="Be verbose",
        action="store_const", dest="loglevel", const=logging.INFO,
    )
    args = parser.parse_args()
    print( args.demo ) 
    logging.basicConfig(level=args.loglevel)
    if args.demo in CONFIGS.keys(): 
        logging.debug('Only shown in debug mode')
        train_ESN( **CONFIGS['QUASIGEO_DEMO']['TRAIN'] )
    else: 
        pass
    
