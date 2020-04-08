import pickle 
import os 
import sys 
import shutil 
import numpy as np 
import matplotlib.pyplot as plt 

curr_fullpath = os.getcwd()
ML_Fluid_fullpath = os.path.abspath("../ML_Fluid/src/")
QG_FTLE_fullpath = os.path.abspath("../QG_FTLE/src/")
ML_Fluid_RESULTS_fullpath = os.path.abspath("../ML_Fluid/results/")
QG_FTLE_INPUTS_fullpath = os.path.abspath("../QG_FTLE/inputs/")
ML_Fluid_raw_inputs_fullpath = os.path.abspath("../ML_Fluid/inputs/raw/")

# DIRTY HACK, list all overlapping modules (by name) in the two dirs
OVERLAPPING_MODULES = ['config', 'util']  

"""
Some notes regarding functions switch_to_qgftle_src_dir() and switch_to_mlfluids_src_dir(): 

It is necessary to change the directory to import the python modules in the relevant directory
(e.g. QG_FTLE or ML_Fluids); however, it is **ALSO** necessary to insert the src to the top of the
path -- this is because there may be modules which share names (e.g. configs.py) across both QG_FTLE
and ML_Fluids src directories, so the namespace must be explicit and the correct src dir must be 
at the top of the path with highest precedence. We must delete the modules that share names
"""

def _switch_to_dir(fullpath): 
    os.chdir(fullpath)
    sys.path.insert(0,fullpath)
    # delete overlapping modules with same name
    for module_name in OVERLAPPING_MODULES: 
        try:
            del sys.modules[module_name]
        except: 
            pass 

def switch_to_qgftle_src_dir(): 
    _switch_to_dir(QG_FTLE_fullpath)

def switch_to_home_dir(): 
    _switch_to_dir(curr_fullpath)

def switch_to_mlfluids_src_dir(): 
    _switch_to_dir(ML_Fluid_fullpath)


def contour_cross_section(data, i, title, fill=False, **kwargs):
    # plot contour graph of the ith cross section of a 3d array
    # for data of shape (x,y,t), plot (x,y) when t=i
    x = np.arange(0.0, data.shape[0], 1)
    y = np.arange(0.0, data.shape[1], 1)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(**kwargs)    

    if fill: 
        CS = ax.contourf(X, Y, data[:,:,i].transpose())
    else: 
        CS = ax.contour(X, Y, data[:,:,i].transpose())
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title(title)
    return fig, ax



def visualize_time_series(iters, data, diff_flag=True):
    """
    Plot two FTLE fields side by side for comparison/visualization 
    Usually *.est and *.actual
    data are 3d numpy arrays
    e.g. something like dgsf.128.64.est and dgsf.128.64.actual
    diff_flag adds another plot to show the diff
    """
    fig = plt.figure()
    num_plots = len(data)+2 if diff_flag else len(data)
    subplot_format = "1"+str(num_plots)
    axs = [ fig.add_subplot(subplot_format+str(i+1)) for  i in range(num_plots) ]
    mywriter = animation.FFMpegWriter()
    ims = []
    ssi_list = []
    mse_list = [] 
    # load FTLE data, 100 frames
    for n in range(iters):
        n_ims = [] 
        for i,dat in enumerate(data):
            n_ims.append(axs[i].imshow(dat, animated=True))
        if diff_flag: 
            # calculate mse
            mse = data[0]**2-data[1]**2
            mse_list.append(mse)
            # calculate ssims
            ssi, ssi_im = ssim(data[0],data[1],use_sample_covariance=False,gaussian_weights=True,full=True)
            ssi_list.append(ssi)
            n_ims.append( axs[num_plots-2].imshow( mse , animated=True) )
            n_ims.append( axs[num_plots-1].imshow( ssi_im , animated=True) )
        ims.append(n_ims)


    avg_ssi = 1.0*sum(ssi_list)/len(ssi_list)
    avg_mse = 1.0*sum(mse_list)/len(mse_list)
    plt.title( 'ssi: {0:.5f} mse: {0:.5f}'.format(avg_ssi, avg_mse) )
    ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True,
                                    repeat_delay=1)        
    
    # ftle_animation_fullpath = os.path.join( RESULT_PATH_DIR, ftle_animation_filename )
    # ani.save( ftle_animation_fullpath )# ,writer = mywriter)

    # plt.plot(ssi_list)
    return avg_ssi

