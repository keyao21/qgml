import os 
from src.config import * 

QG_CONFIG_KEYS = ['QUASIGEO_DEMO']

os.chdir("src")
for CONFIG_KEY in QG_CONFIG_KEYS: 
	os.system( "python preprocess.py --demo {0}".format(CONFIG_KEY) )
	# os.system( "python generate_FTLE_mapping.py --demo {0}".format(CONFIG_KEY) )
	# os.system( "python generate_FTLE_fields.py --demo {0}".format(CONFIG_KEY) )
	
	