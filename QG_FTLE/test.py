import os 

os.chdir("src")
for CONFIG_KEY in ['TESTING']:
	os.system( "python generate_streamfunction_values.py --demo {0}".format(CONFIG_KEY) )
	os.system( "python generate_velocity_fields.py --demo {0}".format(CONFIG_KEY) )
	os.system( "python generate_FTLE_mapping.py --demo {0}".format(CONFIG_KEY) )
	os.system( "python generate_FTLE_fields.py --demo {0}".format(CONFIG_KEY) )
	