"""
Module for general utility functions
"""
import os, shutil
import glob
import pickle
import numpy as np
from fractions import Fraction 

def load_sf_field( sf_fullpath ): 
    # load pickled streamfunction file 
    print('loading sf from ', sf_fullpath, '...')
    with open( sf_fullpath, 'rb' ) as sf_file: 
        sf = pickle.load( sf_file ) 
    print('loaded sf!')
    print('shape', sf.shape)
    return sf 

def save_velocity_field(u, v, uv_fullpath): 
    print('Saving velocity fields to ', uv_fullpath, '...')
    with open( uv_fullpath, 'wb') as uv_file: 
        pickle.dump( (u,v), uv_file )
    print('Saved velcotiy fields')

def load_velocity_field(uv_fullpath ): 
    print('Loading velcotiy fields from ', uv_fullpath, '...')
    with open( uv_fullpath, 'rb' ) as uv_file: 
        u, v = pickle.load( uv_file ) 
    print('loaded uvsf!')
    # print('shape', np.shape(u), np.shape(v))
    if (type(u)==list): 
        print('shape', len(u), len(v))
    else: 
        print('shape', u.shape, v.shape)
    return u, v

def load_FTLE_mapping( mapping_file_fullpath ): 
    print('Loading FTLE mapping files from ', mapping_file_fullpath )
    with open(mapping_file_fullpath , 'rb', ) as map_file: 
        X,Y = np.loadtxt(map_file,unpack=True)
    print('loaded FTLE mapping!')
    print('shape', X.shape, Y.shape)        
    return X,Y

def load_FTLE_field( FTLE_field_file_fullpath ): 
    print('Loading FTLE field files from ', FTLE_field_file_fullpath )
    with open(FTLE_field_file_fullpath , 'rb', ) as ftle_field_file: 
        FTLE = np.loadtxt(ftle_field_file,unpack=True)
    print('loaded FTLE field!')
    print('shape', FTLE.shape)        
    return FTLE

def load_config_dict( config_fullpath ): 
    # load pickled config file 
    print('loading configs from ', config_fullpath, '...')
    with open( config_fullpath, 'rb' ) as config_file: 
        configs = pickle.load( config_file ) 
    print('loaded configs!')
    return configs 

def calculate_xct_yct_ratio( xct, yct ): 
    """
    Return numerator and denominator of reduced fraction xct/yct 
    """
    reduced_fraction = Fraction( xct, yct )
    return reduced_fraction.numerator, reduced_fraction.denominator

def set_up_dir(path_dir):
    """
    Create new directory if directory does not exist, else ignore
    """
    try: 
        os.makedirs(path_dir)
        print("Created new directory ", path_dir)
    except FileExistsError: 
        pass

def reset_dir(path_dir): 
    """
    Create new directory if directory does not exist, else wipe contents of directory
    """
    try: 
        os.makedirs(path_dir)
        print("Created new directory ", path_dir)
    except FileExistsError: 
        filelist = glob.glob(os.path.join(path_dir, "mapping*"))
        for f in filelist:
            os.remove(f)
        # shutil.rmtree(path_dir)
        # os.makedirs(path_dir)
        print("Reset directory ", path_dir)


def shift_numpy_array(arr, num, fill_value=np.nan):
    """
    Shift numpy array arr elements by num positions
    ref: https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
    """
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def add_numpy_arrays(a,b):
    """
    Add two numpy arrays of different lenghts 
    ref: https://stackoverflow.com/questions/7891697/numpy-adding-two-vectors-with-different-sizes
    """
    if len(a) < len(b):
        c = b.copy()
        c[:len(a)] += a
    else:
        c = a.copy()
        c[:len(b)] += b
    return c


def moving_average_smooth(y, box_pts):
    """
    smooth lines via moving average box 
    """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def stack_numpy_arrays(a,b): 
    return np.vstack([a,b])
    

