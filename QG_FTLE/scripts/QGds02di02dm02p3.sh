# conda activate
cd ../src
CONFIG_KEYS='QGds02di02dm02p3.2000.1p8.actual QGds02di02dm02p3.2000.1p8.est'
for CONFIG_KEY in $CONFIG_KEYS 
do  
	python generate_velocity_fields.py --demo $CONFIG_KEY
	python generate_FTLE_mapping.py --demo $CONFIG_KEY
	python generate_FTLE_fields.py --demo $CONFIG_KEY
done 
python compare_FTLE_fields.py --iters 20 --ftle_path_dirs $CONFIG_KEY --ftle_animation_filename QGds02di02dm02p3.gif
read -rsp $'End of script: press any key to continue...\n' -n1 key




