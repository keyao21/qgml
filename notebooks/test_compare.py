import nb_utils
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

nb_utils.switch_to_qgftle_src_dir()
import util




sfactual_filename = "dgsf_0p1_128_64_0p1_0p21000_2.0.actual"
sfactual_fullpath = os.path.join(nb_utils.QG_FTLE_INPUTS_fullpath,sfactual_filename)
sfactual = util.load_sf_field(sf_fullpath=sfactual_fullpath)

sfest_filename = "dgsf_0p1_128_64_0p1_0p21000_2.0.est"
sfest_fullpath = os.path.join(nb_utils.QG_FTLE_INPUTS_fullpath,sfest_filename)
sfest = util.load_sf_field(sf_fullpath=sfest_fullpath)




iters, data, diff_flag = 1000, [sfactual,sfest], False
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
        n_ims.append(axs[i].imshow(dat[:,:,n], animated=True))
    if diff_flag: 
        # calculate mse
        mse = data[0][:,:,n]**2-data[1][:,:,n]**2
        mse_list.append(mse)
        # calculate ssims
        ssi, ssi_im = ssim(data[0][:,:,n],data[1][:,:,n],use_sample_covariance=False,gaussian_weights=True,full=True)
        ssi_list.append(ssi)
        n_ims.append( axs[num_plots-2].imshow( mse , animated=True) )
        n_ims.append( axs[num_plots-1].imshow( ssi_im , animated=True) )
    ims.append(n_ims)

if diff_flag:
    avg_ssi = 1.0*sum(ssi_list)/len(ssi_list)
    avg_mse = 1.0*sum(mse_list)/len(mse_list)
    plt.title( 'ssi: {0:.5f} mse: {0:.5f}'.format(avg_ssi, avg_mse) )
ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True,repeat_delay=1)   

nb_utils.switch_to_home_dir()                        
import pdb;pdb.set_trace()
ani.save( "test.gif" )# ,writer = mywriter)
# ftle_animation_fullpath = os.path.join( RESULT_PATH_DIR, ftle_animation_filename )

# plt.plot(ssi_list)
# return avg_ssi

