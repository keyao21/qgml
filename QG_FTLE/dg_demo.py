import os 

os.chdir("src")
# for CONFIG_KEY in ['DOUBLE_GYRE_ACTUAL', 'DOUBLE_GYRE_EST']: 
for CONFIG_KEY in ['DOUBLE_GYRE_EST']: 
	os.system( "python generate_velocity_fields.py --demo {0}".format(CONFIG_KEY) )
	os.system( "python generate_FTLE_mapping.py --demo {0}".format(CONFIG_KEY) )
	os.system( "python generate_FTLE_fields.py --demo {0}".format(CONFIG_KEY) )
	