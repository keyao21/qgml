import time 
import argparse 
import os 
from config import * 
import util 
import interp 
import numpy as np 

def generate_mapping_files(iters, mapped_dt, dt, xct, yct, velocity_func_filename, mapping_path_dir): 
    """
    Generate FTLE mapping files named "mapping<int>.txt" in directory

    iters: set number of mapping files (bounded by max_mapping_file_ct)
    mapped_dt: time step for FTLE mapping 
    dt: time step for RK4 procedure (should match with time step used to calculate stream function series)
    xct, yct: number of cells in field dimensions
    mapping_path_dir: file directory which will contain mapping files 
    """
    mapping_dir_fullpath = os.path.join( FTLE_MAPPING_PATH_DIR, mapping_path_dir)
    print('Generating mapping files in ', mapping_dir_fullpath ,'...')
    util.reset_dir(mapping_dir_fullpath)

    uvinterp_fullpath = os.path.join( INTERP_VELOCITY_PATH_DIR, velocity_func_filename)
    vfuncs = util.load_velocity_field(uvinterp_fullpath)
    
    start_time = time.time()
    reduced_xct, reduced_yct = util.calculate_xct_yct_ratio( xct, yct )
    # shift = (1.0/xct)*(0.5)  # middle of first cell 
    # dx and dy should be equal 
    shift = float(min( reduced_xct, reduced_yct )) / (2*float( min( xct, yct ) ))

    # y-coordinate of the grid
    y_coords = np.linspace(shift,reduced_yct-shift,yct)
    # evolution time 0-10s, max_mapping_file_ct time steps maximum
    max_mapping_file_ct = 1000000
    tau_range = np.linspace(0.0,dt*(max_mapping_file_ct-1),max_mapping_file_ct)
    # x and y ranges for interp.velcoity_update() 
    x_range, y_range = (0.0, float(reduced_xct)), (0.0, float(reduced_yct) )
    # getting 100 mapping data files
    for n,tau_n in enumerate(tau_range):
        if n == iters: break
        mapping_file_fullpath = os.path.join(mapping_dir_fullpath, 'mapping{0}'.format(n) )
        print("Saving file: ", mapping_file_fullpath )
        output = open(  mapping_file_fullpath ,'ab')
        for y_i in y_coords:
            # initialize grid horizontally
            x = np.linspace(shift, reduced_xct-shift, xct)
            y = np.linspace(y_i, y_i, xct)
            r = np.column_stack( (x,y) )
            # perform RK4 to get position of particle 20s later
            for t in np.arange(0+tau_n,mapped_dt+tau_n,dt):
                k1 = dt*interp.velocity_update(vfuncs, r, t, dt, x_range, y_range )
                k2 = dt*interp.velocity_update(vfuncs, r+0.5*k1, t+0.5*dt, dt, x_range, y_range )
                k3 = dt*interp.velocity_update(vfuncs, r+0.5*k2, t+0.5*dt, dt, x_range, y_range )
                k4 = dt*interp.velocity_update(vfuncs, r+k3, t+dt, dt, x_range, y_range )
                r += (k1+2*k2+2*k3+k4)/6
            # append data to the file   
            np.savetxt(output,r)       
        output.close()
    print(time.time()-start_time)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', choices=CONFIGS.keys(), help='Keys from CONFIGS dict in config.py ')
    args = parser.parse_args()
    print( args.demo ) 

    if args.demo in CONFIGS.keys(): 
        generate_mapping_files( **CONFIGS[ args.demo ]['GENERATE_FTLE_MAPPING'] )
    else: 
        iters = float(input("Iterations of mapping (# FTLE frames): "))
        generate_mapping_files(  )


