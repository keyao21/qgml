import os 

INPUT_PATH_DIR = "../inputs"
RAW_INPUT_PATH_DIR = os.path.join( INPUT_PATH_DIR, 'raw')
PREPROCESS_INPUT_PATH_DIR = os.path.join( INPUT_PATH_DIR, 'preprocess')

DATA_PATH_DIR = "../data"
RESULTS_PATH_DIR = "../results"

CONFIGS = { 
	'QUASIGEO_DEMO': {
		'PREPROCESS' : { 
			'MATLAB_filename' : 'QG_psi_ds0.02_di0.02_dm0.02_pertamp0.3.mat', 
			'training_length'		  : 1000,
			'training_data_filename'  : 'QGds02di02dm02p3.TRAIN',
			'testing_data_filename'  : 'QGds02di02dm02p3.TEST'	
		},
		'TRAIN' : { 
			'training_data_filename'  : 'QGds02di02dm02p3.TRAIN',
			'trained_model_params'    : { 
	            "initLen"           : 0, 
	            "resSize"           : 1000, 
	            "partial_know"      : False, 
	            "noise"             : 1e-2, 
	            "density"           : 1e-3, 
	            "spectral_radius"   : 1.8, 
	            "leaking_rate"      : 0.2, 
	            "input_scaling"     : 0.3, 
	            "ridgeReg"          : 0.01, 
	            "mute"              : False 
			},
			'trained_model_filename'  : 'QGds02di02dm02p3.ESN',
		},
		'TEST' : { 
			'testing_data_filename'   : 'QGds02di02dm02p3.TEST',
			'testing_length'		  : 1000,
			'trained_model_filename'  : 'QGds02di02dm02p3.ESN', 
			'output_filenames'		  : {
				'stream_function_estimated' : 'QGds02di02dm02p3.est',
				'stream_function_actual'    : 'QGds02di02dm02p3.actual',
			}

		}
	}
}