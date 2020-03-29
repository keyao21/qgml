# import os 
# os.chdir("../src")
# from config import * 

# QG_CONFIG_KEYS = ['QGds02di02dm02p3.1000.1p8', \
# 				'QGds02di02dm02p3.750.1p8',\
# 				'QGds02di02dm02p3.250.1p8']

# for CONFIG_KEY in QG_CONFIG_KEYS: 
#     os.system( "python preprocess.py --demo {0}".format(CONFIG_KEY) )
#     os.system( "python train.py --demo {0}".format(CONFIG_KEY) )
#     os.system( "python test.py --demo {0}".format(CONFIG_KEY) )

cd ../src
CONFIG_KEYS='QGds02di02dm02p3.1000.0p3'
 # QGds02di02dm02p3.100.1p8 QGds02di02dm02p3.1250.1p8 QGds02di02dm02p3.2000.1p8'
for CONFIG_KEY in $CONFIG_KEYS 
do  
	python preprocess.py --demo $CONFIG_KEY
	python train.py --demo $CONFIG_KEY
	python test.py --demo $CONFIG_KEY
done 
# python compare_FTLE_fields.py --iters 20 --ftle_path_dirs $CONFIG_KEY --ftle_animation_filename QGds02di02dm02p3.gif
read -rsp $'End of script: press any key to continue...\n' -n1 key




