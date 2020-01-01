import argparse 
import os 
from config import * 
import interp
import util 

def generate_velocity_fields( stream_function_filename, velocity_filename, velocity_func_filename ): 
    """
    Loads in stream function file, generates and saves velocity field (both discrete and interp files
    stream_function_filename: string 
    velocity_filename: string
    velocity_func_filename: string
    """
    sf_fullpath = os.path.join( INPUT_PATH_DIR, stream_function_filename )
    uv_fullpath = os.path.join( DISCRETE_VELOCITY_PATH_DIR, velocity_filename )
    uvinterp_fullpath = os.path.join( INTERP_VELOCITY_PATH_DIR, velocity_func_filename )

    sf = util.load_sf_field( sf_fullpath=sf_fullpath )
    u, v = interp.calculate_velocity_field( sf )
    util.save_velocity_field( u, v, uv_fullpath=uv_fullpath)
    uinterp, vinterp = interp.calculate_interp_velocity_funcs(u, v)
    util.save_velocity_field( uinterp, vinterp, uv_fullpath=uvinterp_fullpath)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', choices=CONFIGS.keys(), help='Keys from CONFIGS dict in config.py ')
    args = parser.parse_args()
    print( args.demo ) 

    if args.demo in CONFIGS.keys(): 
        generate_velocity_fields( **CONFIGS[ args.demo ]['GENERATE_VELOCITY_FIELDS'] )
    else: 
        stream_function_filename = str(input("Stream function filename: "))
        velocity_filename = str(input("Discrete velocity field filename: "))
        velocity_func_filename = str(input("Interpolated velocity function filename: "))
        generate_velocity_fields( stream_function_filename, velocity_filename, velocity_func_filename )

