import argparse 
import os 
import numpy as np 
import matplotlib.pyplot as plt
# plt.rcParams['image.cmap'] = 'PuOr'
# plt.rcParams['image.cmap'] = 'cool'
# plt.rcParams['image.cmap'] = 'gnuplot'
# plt.rcParams['image.cmap'] = 'Spectral'
plt.rcParams['image.cmap'] = 'cividis'

import matplotlib.animation as animation
from config import * 

def compare_FTLE_animation(iters, ftle_path_dirs, ftle_animation_filename, diff_flag=True):
    """
    Plot two FTLE fields side by side for comparison 
    Usually *.est and *.actual
    ftle_path_dirs is a list type, where entries are directory name strings 
    e.g. ( 'dgsf.128.64.est', 'dgsf.128.64.actual' )
    diff_flag adds another plot to show the diff
    """
    fig = plt.figure()
    num_plots = len(ftle_path_dirs)+1 if diff_flag else len(ftle_path_dirs)
    subplot_format = "1"+str(num_plots)
    axs = [ fig.add_subplot(subplot_format+str(i+1)) for  i in range(num_plots) ]
    FTLE_dir_fullpaths = [ os.path.join( FTLE_FIELDS_PATH_DIR, ftle_path_dir ) for ftle_path_dir in ftle_path_dirs ]
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
        n_ims = [] 
        ftle_field_file_fullpaths = [ os.path.join( FTLE_dir_fullpath, 'FTLE{0}'.format(n) ) for FTLE_dir_fullpath in FTLE_dir_fullpaths ]
        FTLEs = [ np.loadtxt(ftle_field_file_fullpath) for ftle_field_file_fullpath in ftle_field_file_fullpaths ]  
        for i,FTLE in enumerate(FTLEs):
            n_ims.append(axs[i].imshow(FTLE, animated=True))
        if diff_flag: 
            n_ims.append( axs[num_plots-1].imshow(FTLEs[num_plots-2]-FTLEs[num_plots-3], animated=True) )
        ims.append(n_ims)
    ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True,
                                    repeat_delay=1)        
    
    # plt.colorbar()                              
    # plt.gca().invert_yaxis()
    # make a video
    plt.show()

    ftle_animation_fullpath = os.path.join( RESULT_PATH_DIR, ftle_animation_filename )
    ani.save( ftle_animation_fullpath )# ,writer = mywriter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', '-i', type=int)
    parser.add_argument('--ftle_path_dirs', '-p', nargs='+')
    parser.add_argument('--ftle_animation_filename', '-f')

    # parser.add_argument('string', help='Input String', nargs='+')
    # args = parser.parse_args()
    # arg_str = ' '.join(args.string)
    # print(arg_str)

    args = parser.parse_args()
    iters = args.iters 
    ftle_path_dirs = args.ftle_path_dirs
    ftle_animation_filename = args.ftle_animation_filename

    print( args.iters ) 
    print( args.ftle_path_dirs)
    print( args.ftle_animation_filename )
    
    if None not in [iters, ftle_path_dirs, ftle_animation_filename]: 
        compare_FTLE_animation( iters, ftle_path_dirs, ftle_animation_filename )
    else: 
        iters = float(input("Iterations of FTLE fields (# FTLE frames): "))
        compare_FTLE_animation(  )
