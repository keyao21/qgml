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
from skimage.measure import compare_ssim as ssim
# from skimage.metrics import mean_squared_error as calc_mse
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
    num_plots = len(ftle_path_dirs)+2 if diff_flag else len(ftle_path_dirs)
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
    ssi_list = []
    mse_list = [] 
    # load FTLE data, 100 frames
    for n in range(iters):
        n_ims = [] 
        ftle_field_file_fullpaths = [ os.path.join( FTLE_dir_fullpath, 'FTLE{0}'.format(n) ) for FTLE_dir_fullpath in FTLE_dir_fullpaths ]
        FTLEs = [ np.loadtxt(ftle_field_file_fullpath) for ftle_field_file_fullpath in ftle_field_file_fullpaths ]  
        for i,FTLE in enumerate(FTLEs):
            n_ims.append(axs[i].imshow(FTLE, animated=True))
        if diff_flag: 
            # calculate mse
            mse = FTLEs[0]**2-FTLEs[1]**2
            # mse = calc_mse( FTLEs[0],FTLEs[1] )
            mse_list.append(mse)
            # calculate ssims
            ssi, ssi_im = ssim(FTLEs[0],FTLEs[1],use_sample_covariance=False,gaussian_weights=True,full=True)
            ssi_list.append(ssi)

            n_ims.append( axs[num_plots-2].imshow( mse , animated=True) )
            n_ims.append( axs[num_plots-1].imshow( ssi_im , animated=True) )

        # title = axs[num_plots-1].text(32,-10,"SSI: {0:.2f}%".format(ssi*100) )
        # n_ims.append(title)
        ims.append(n_ims)


    avg_ssi = 1.0*sum(ssi_list)/len(ssi_list)
    avg_mse = 1.0*sum(mse_list)/len(mse_list)
    plt.title( 'ssi: {0:.5f} mse: {0:.5f}'.format(avg_ssi, avg_mse) )
    ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True,
                                    repeat_delay=1)        
    
    # plt.colorbar()                              
    # plt.gca().invert_yaxis()
    # make a video
    plt.show()

    ftle_animation_fullpath = os.path.join( RESULT_PATH_DIR, ftle_animation_filename )
    ani.save( ftle_animation_fullpath )# ,writer = mywriter)


    # print('hi') 
    plt.plot(ssi_list)
    plt.show()
    # input()

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
        iters = 20
        # ftle_path_dirs = ['qgsf.64.128.actual','qgsf.64.128.est']
        # ftle_path_dirs = ['dgsf.128.64.actual','dgsf.128.64.est']
        # ftle_path_dirs = ['dgsf_0p01_200_128_64_0p1_0p2.actual','dgsf_0p01_200_128_64_0p1_0p2.est']
        ftle_path_dirs = ['QGds02di02dm02p3.actual','QGds02di02dm02p3.est']
        ftle_animation_filename = 'QGds02di02dm02p3_compare.gif'
        compare_FTLE_animation( iters, ftle_path_dirs, ftle_animation_filename )

