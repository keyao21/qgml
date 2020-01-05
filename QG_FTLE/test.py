import os 
import argparse 
from src.config import * 

def run_demo( CONFIG_KEY ): 
    os.chdir("src")
    os.system( "python double_gyre.py --demo {0}".format(CONFIG_KEY) )
    os.system( "python generate_velocity_fields.py --demo {0}".format(CONFIG_KEY) )
    os.system( "python generate_FTLE_mapping.py --demo {0}".format(CONFIG_KEY) )
    os.system( "python generate_FTLE_fields.py --demo {0}".format(CONFIG_KEY) )
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', choices=CONFIGS.keys(), help='Keys from CONFIGS dict in config.py ', nargs='+')
    args = parser.parse_args()
    print( args.demo ) 

    for config_key in args.demo:
        if config_key in CONFIGS.keys(): 
            run_demo( config_key )
        



