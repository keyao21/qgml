# conda activate
cd ../src
CONFIG_KEYS_1000_1p8='QGds02di02dm02p3.1000.1p8.actual QGds02di02dm02p3.1000.1p8.est'
CONFIG_KEYS_1250_1p8='QGds02di02dm02p3.1250.1p8.actual QGds02di02dm02p3.1250.1p8.est'
CONFIG_KEYS_2000_1p8='QGds02di02dm02p3.2000.1p8.actual QGds02di02dm02p3.2000.1p8.est'


python compare_FTLE_fields.py --iters 20 --ftle_path_dirs $CONFIG_KEYS_1000_1p8 --ftle_animation_filename QGds02di02dm02p3.1000.1p8..gif
python compare_FTLE_fields.py --iters 20 --ftle_path_dirs $CONFIG_KEYS_1250_1p8 --ftle_animation_filename QGds02di02dm02p3.1250.1p8.gif
python compare_FTLE_fields.py --iters 20 --ftle_path_dirs $CONFIG_KEYS_2000_1p8 --ftle_animation_filename QGds02di02dm02p3.2000.1p8.gif
read -rsp $'End of script: press any key to continue...\n' -n1 key




