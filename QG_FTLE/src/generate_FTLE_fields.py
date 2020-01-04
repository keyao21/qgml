import time 
import argparse 
import os 
from config import * 
import util 
import interp 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def Jacobian(X, Y, xct, yct):      
    # spatial Jacobian that is used to compute FTLE
    # calculate dx and dy 
    reduced_xct, reduced_yct = util.calculate_xct_yct_ratio( xct, yct )
    dx = float(reduced_xct)/float(xct)
    dy = float(reduced_yct)/float(yct)
    J = np.empty([2,2],float)
    FTLE = np.empty([yct-2,xct-2],float)
    for i in range(0,yct-2):
        for j in range(0,xct-2):
            J[0][0] = (X[(1+i)*xct+2+j]-X[(1+i)*xct+j])/(2*dx)
            J[0][1] = (X[(2+i)*xct+1+j]-X[i*xct+1+j])/(2*dx)
            J[1][0] = (Y[(1+i)*xct+2+j]-Y[(1+i)*xct+j])/(2*dy)
            J[1][1] = (Y[(2+i)*xct+1+j]-Y[i*xct+1+j])/(2*dy)
            # Green-Cauchy tensor
            D = np.dot(np.transpose(J),J)
            # import pdb;pdb.set_trace()
            # its largest eigenvalue
            lamda = np.linalg.eigvals(D)
            FTLE[i][j] = max(lamda)
    return FTLE

def generate_FTLE_fields(iters, xct, yct, mapping_path_dir, ftle_path_dir): 
    start_time = time.time()
    mapping_dir_fullpath = os.path.join( FTLE_MAPPING_PATH_DIR, mapping_path_dir)
    FTLE_dir_fullpath = os.path.join( FTLE_FIELDS_PATH_DIR, ftle_path_dir )
    util.reset_dir(FTLE_dir_fullpath)

    print('Loading mapping files from ', mapping_dir_fullpath ,'...')
    for n in range(iters):
        mapping_file_fullpath = os.path.join( mapping_dir_fullpath,  'mapping{0}'.format(n) )
        X,Y = util.load_FTLE_mapping( mapping_file_fullpath )
        FTLE = Jacobian(X, Y, xct, yct)
        FTLE = np.log(FTLE)

        ftle_field_file_fullpath = os.path.join( FTLE_dir_fullpath, 'FTLE{0}'.format(n) )
        print('Saving FTLE field files to ', ftle_field_file_fullpath)
        np.savetxt(ftle_field_file_fullpath,FTLE)
    print(time.time()-start_time)

def generate_FTLE_animation(iters, xct, yct, ftle_path_dir, ftle_animation_filename):
    fig = plt.figure()
    FTLE_dir_fullpath = os.path.join( FTLE_FIELDS_PATH_DIR, ftle_path_dir )
    
    # movie_maker
    #plt.rcParams['animation.ffmpeg_path'] = 'C:/ffmpeg/bin/ffmpeg'
    mywriter = animation.FFMpegWriter()
    # rescale ticks
    # reduced_xct, reduced_yct = util.calculate_xct_yct_ratio( xct, yct )
    # x = [i for i in range(0,xct)][::int(xct/5.0)]
    # y = [i for i in range(0,yct)][::int(yct/5.0)]

    # xlabels = [i for i in range(0,xct)][::int(reduced_xct/5.0)]
    # ylabels = [i for i in range(0,yct)][::int(reduced_yct/5.0)]
    
    # # xlabels = ['0','0.5','1.0','1.5','2.0']
    # # ylabels = ['0','0.5','1.0']
    # plt.xticks(x,xlabels)
    # plt.yticks(y,ylabels)
    ims = []
    # load FTLE data, 100 frames
    for n in range(iters):
        ftle_field_file_fullpath = os.path.join( FTLE_dir_fullpath, 'FTLE{0}'.format(n) )
        F = np.loadtxt(ftle_field_file_fullpath) 
        im = plt.imshow(F, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True,
                                    repeat_delay=1)        
    plt.colorbar()                              
    plt.gca().invert_yaxis()
    # make a video

    ftle_animation_fullpath = os.path.join( RESULT_PATH_DIR, ftle_animation_filename )
    # ani.save( ftle_animation_fullpath )# ,writer = mywriter)
    plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', choices=CONFIGS.keys(), help='Keys from CONFIGS dict in config.py ')
    args = parser.parse_args()
    print( args.demo ) 

    if args.demo in CONFIGS.keys(): 
        generate_FTLE_fields( **CONFIGS[ args.demo ]['GENERATE_FTLE_FIELDS'] )
        generate_FTLE_animation( **CONFIGS[ args.demo ]['GENERATE_FTLE_ANIMATIONS'] )
    else: 
        iters = float(input("Iterations of FTLE fields (# FTLE frames): "))
        generate_FTLE_fields(  )



