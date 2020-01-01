import os 
from src.config import * 

QG_CONFIG_KEYS = ['QUASIGEO_ACTUAL', 'QUASIGEO_EST']
FTLE_PATH_DIRS = [
	CONFIGS[ QG_CONFIG_KEY ]['GENERATE_FTLE_FIELDS']['ftle_path_dir'] 
	for QG_CONFIG_KEY in QG_CONFIG_KEYS
]

os.chdir("src")
for CONFIG_KEY in QG_CONFIG_KEYS: 
	os.system( "python generate_velocity_fields.py --demo {0}".format(CONFIG_KEY) )
	os.system( "python generate_FTLE_mapping.py --demo {0}".format(CONFIG_KEY) )
	os.system( "python generate_FTLE_fields.py --demo {0}".format(CONFIG_KEY) )
	
	