import util 
from config import *  
import os 
import argparse 

def preprocess_input_data( MATLAB_filename, training_length, training_data_filename, testing_data_filename ):
    """
    This function's purpose is to load the raw input stream function time series data and 
    save down as numpy array, split into training and testing segments
    """
    MATLAB_file_fullpath = os.path.join( RAW_INPUT_PATH_DIR, MATLAB_filename) 
    preprocessed_training_file_fullpath = os.path.join( PREPROCESS_INPUT_PATH_DIR, training_data_filename )
    preprocessed_testing_file_fullpath = os.path.join( PREPROCESS_INPUT_PATH_DIR, testing_data_filename )
    data = util.load_mat_file( mat_fullpath=MATLAB_file_fullpath, var_name='Psi_ts' )
    # adding +1 to training length because we need (training_length) pairs of consecutive time steps
    train_data, test_data = util.split_training_testing( data, training_length=training_length+1, axis=2)
    util.save_data( train_data, preprocessed_training_file_fullpath )
    util.save_data( test_data, preprocessed_testing_file_fullpath )
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', choices=CONFIGS.keys(), help='Keys from CONFIGS dict in config.py ')
    args = parser.parse_args()
    print( args.demo ) 
    if args.demo in CONFIGS.keys(): 
        preprocess_input_data( **CONFIGS[ args.demo ]['PREPROCESS'] )
    else: 
        pass
    