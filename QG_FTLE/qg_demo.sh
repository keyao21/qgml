cd src
CONFIG_KEYS='QUASIGEO_ACTUAL QUASIGEO_EST'
for CONFIG_KEY in $CONFIG_KEYS 
do  
	python generate_velocity_fields.py --demo $CONFIG_KEY
	python generate_FTLE_mapping.py --demo $CONFIG_KEY
	python generate_FTLE_fields.py --demo $CONFIG_KEY
done 

