"""
MOdule for outputing plots comparing trajectories
"""
import argparse 
import os 
import numpy as np 
import matplotlib.pyplot as plt
import util 
import double_gyre as dg
import interp 
from config import * 


def generate_single_trajectory( velocity_func_fullpaths, initial_conditions, 
                                elapsed_time, dt,  ):
    """
    Create trajectory plots for single intiial condition
    Loads stream function file and calculate trajectories
    using RK4
    
    velocity_func_filename: list of string filename(s)
    initial_conditions: tuple of initial condition coordinates
    """
    num_samples = 1
    vfuncs_list = [] 
    for velocity_func_fullpath in velocity_func_fullpaths:
        vfuncs_list.append( util.load_velocity_field(velocity_func_fullpath) )
        # TODO: MULTIPLE VFUNCS PER VELOCITY_FUNC_FULLPATH
        #!!!!

    # Generate estimated trajectories
    # state : numpy array shaped as ...
    # ( num velocity_funcs, num_samples, time steps, num coordinates)
    state = np.zeros(( len(velocity_func_fullpaths), num_samples, int(elapsed_time/dt)+1, 2 ))
    for vfunc_idx, vfuncs in enumerate(vfuncs_list):
        state[vfunc_idx,:,0,0] = initial_conditions[0]
        state[vfunc_idx,:,0,1] = initial_conditions[1]
        for i,t in enumerate(np.arange(0, elapsed_time, dt)):
            state[vfunc_idx,:,i+1,:] = interp.rk4(vfuncs, state[vfunc_idx,:,i,:].copy(), t, dt)
        print('Trajectories finished.')

    plt.figure()
    plt.subplot(2,1,1)
    for i,_ in enumerate(vfuncs_list):
        plt.plot(state[i,0,:,0].transpose())

    plt.subplot(2,1,2)
    for i,_ in enumerate(vfuncs_list):
        plt.plot(state[i,0,:,1].transpose())
    plt.show()

    return state




def compare_single_trajectories(  ): 
    pass
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj', '--trajectories', '-t', type=int)
    args = parser.parse_args()



