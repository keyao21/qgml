# conda activate
cd ../src
CONFIG_KEYS='dgsf_0p01_200_128_64_0p1_0p2.actual dgsf_0p01_200_128_64_0p1_0p2.est'
for CONFIG_KEY in $CONFIG_KEYS 
do  
	python generate_velocity_fields.py --demo $CONFIG_KEY
	python generate_FTLE_mapping.py --demo $CONFIG_KEY
	python generate_FTLE_fields.py --demo $CONFIG_KEY
done 
python compare_FTLE_fields.py $CONFIG_KEY dgsf_0p01_200_128_64_0p1_0p2_compare.gif

read -rsp $'End of script: press any key to continue...\n' -n1 key




