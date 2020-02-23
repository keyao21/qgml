import numpy as np 
import argparse
import logging
import os
from ESN import EchoStateNetwork
from config import * 
import util


def test_ESN( testing_data_filename, testing_length, trained_model_filename, output_filenames ): 
    """
    Initialize and test echo state network on test data with trained model
    """
    input_data_fullpath = os.path.join( PREPROCESS_INPUT_PATH_DIR, testing_data_filename)
    input_data = util.load_data( input_data_fullpath )
    xct, yct, time_steps = np.shape(input_data) 
    flattened_input_data = util.flatten_time_series( input_data ).transpose()
    esn = util.load_model( os.path.join( DATA_PATH_DIR, trained_model_filename) )
    esn.test( testing_data=flattened_input_data[:testing_length+1] )
    sf_est = esn.v_.reshape(xct, yct, -1)  #x by y by t
    sf_actual = esn.v_tgt_.transpose().reshape(xct, yct, -1)   #x by y by t
    util.save_data(sf_est, os.path.join(RESULTS_PATH_DIR, output_filenames['stream_function_estimated']))
    util.save_data(sf_actual, os.path.join(RESULTS_PATH_DIR, output_filenames['stream_function_actual']))

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
        test_ESN( **CONFIGS[args.demo]['TEST'] )
    else: 
        pass
    
