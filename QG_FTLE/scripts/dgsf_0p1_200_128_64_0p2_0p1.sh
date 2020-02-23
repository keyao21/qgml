# conda activate
cd ../src
CONFIG_KEYS='dgsf_0p1_200_128_64_0p2_0p1.actual dgsf_0p1_200_128_64_0p2_0p1.est'
for CONFIG_KEY in $CONFIG_KEYS 
do  
	python generate_velocity_fields.py --demo $CONFIG_KEY
	python generate_FTLE_mapping.py --demo $CONFIG_KEY
	python generate_FTLE_fields.py --demo $CONFIG_KEY
done 
read -rsp $'End of script: press any key to continue...\n' -n1 key




