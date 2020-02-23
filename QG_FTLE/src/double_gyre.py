import pdb
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import pickle
import util 
import interp
from config import * 

# wave function that defines the characteristics of 
# double gyre
def phi(x,y,t,amp,epsilon):
    # amp = 0.2
    temp = amp*np.sin(np.pi*f(x,t,epsilon))*np.sin(np.pi*y)
    return temp

def f(x,t,epsilon):
    # epsilon = 0.3
    w = np.pi/5
    temp = epsilon*np.sin(w*t)*x**2+(1-2*epsilon*np.sin(w*t))*x
    return temp
  
# function that computes velocity of particle at each point
def update(state,t,delta=0.001,amp=0.2,epsilon=0.3):
    x = state[:,0]
    y = state[:,1]
    vx = (phi(x,y+delta,t,amp,epsilon)-phi(x,y-delta,t,amp,epsilon))/(2*delta)
    vy = (phi(x+delta,y,t,amp,epsilon)-phi(x-delta,y,t,amp,epsilon))/(2*delta)
    return np.column_stack((-vx,vy))

def rk4(state,t,dt=0.1,amp=0.2,epsilon=0.3):
    tmp_state = state[:,0:2]
    k1 = dt*update(tmp_state,t,amp,epsilon)
    k2 = dt*update(tmp_state+0.5*k1,t+0.5*dt,amp,epsilon)
    k3 = dt*update(tmp_state+0.5*k2,t+0.5*dt,amp,epsilon)
    k4 = dt*update(tmp_state+k3,t+dt,amp,epsilon)
    tmp_state += (k1+2*k2+2*k3+k4)/6
    state[:,0] = np.clip(tmp_state[:,0],0.01,2)
    state[:,1] = np.clip(tmp_state[:,1],0.01,1)
    #noise = B*np.random.normal(u,sigma,(L,2))
    #state[:,4:6] += noise
    return state

def generate_streamfunction_values(dt=0.1, elapsedTime=500, 
                                    xct=128, yct=64, amp=0.2,epsilon=0.3,
                                    stream_function_filename=None): 
    # dt = 0.1
    # elapsedTime = 200
    time_steps = int(np.ceil(elapsedTime/dt))

    # dx and dy are equal
    reduced_xct, reduced_yct = util.calculate_xct_yct_ratio( xct , yct )    
    dy = dx = float(min( reduced_xct, reduced_yct )) / (float( min( xct, yct ) ))
    # create meshgrid of actual x, y values 
    # note: these values are the center of the cells, so must be shifted by dx and dy 
    x = np.linspace(0.5*dx,float(reduced_xct)-(0.5*dx),xct )
    y = np.linspace(0.5*dy,float(reduced_yct)-(0.5*dy),yct )
    xv, yv = np.meshgrid(x,y) 
    
    # create meshgrid of INDEX x, y values
    x_i = np.arange(0, xct , 1)
    y_i = np.arange(0, yct , 1)
    xis, yis = np.meshgrid(x_i, y_i)
    # create 3d structure to hold x,y values 
    sf_ts = np.zeros((xct, yct, time_steps))
    for i,t_i in enumerate(np.arange(0,elapsedTime,dt)):
        sf_ts[xis, yis, i] = phi(xv, yv, t_i, amp, epsilon)
    # print(sf_ts)

    saved_streamfunction_dir_fullpath = os.path.join( INPUT_PATH_DIR, stream_function_filename)
    
    with open(saved_streamfunction_dir_fullpath, 'wb') as sf_ts_dir_file: 
        pickle.dump(sf_ts, sf_ts_dir_file)
    return sf_ts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', choices=CONFIGS.keys(), help='Keys from CONFIGS dict in config.py ')
    args = parser.parse_args()
    
    if args.demo in CONFIGS.keys(): 
        sf_ts = generate_streamfunction_values(**CONFIGS[ args.demo ]['GENERATE_STREAM_FUNCTION_FIELDS'] )
    else: 
        dt = float(input("dt: "))
        elapsedTime = float(input("elapsedTime: "))
        xct = int(input("xct: "))
        yct = int(input("yct: "))
        amp = float(input("amp: "))
        epsilon = float(input("epsilon: "))
        stream_function_filename = input("stream_function_filename: ")
        sf_ts = generate_streamfunction_values(dt, elapsedTime, xct, yct, amp, epsilon, stream_function_filename)

    