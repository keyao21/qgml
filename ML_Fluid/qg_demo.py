import os 
from src.config import * 

QG_CONFIG_KEYS = ['QUASIGEO_DEMO']

os.chdir("src")
for CONFIG_KEY in QG_CONFIG_KEYS: 
    os.system( "python preprocess.py --demo {0}".format(CONFIG_KEY) )
    os.system( "python train.py --demo {0}".format(CONFIG_KEY) )
    os.system( "python test.py --demo {0}".format(CONFIG_KEY) )

    