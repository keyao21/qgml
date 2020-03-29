import os 
INPUT_PATH_DIR = os.path.abspath("../inputs")
RAW_INPUT_PATH_DIR = os.path.abspath( '../inputs/raw')
PREPROCESS_INPUT_PATH_DIR = os.path.abspath( '../inputs//preprocess')
DATA_PATH_DIR = os.path.abspath("../data")
RESULTS_PATH_DIR = os.path.abspath("../results")

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
	},

##########################################################################################
# quasigeo experiment configs to test ssi for changing ML params (res size and spec rad)
	'QGds02di02dm02p3.1000.1p8': {
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
	            "density"           : 1e-2, 
	            "spectral_radius"   : 1.8, 
	            "leaking_rate"      : 0.2, 
	            "input_scaling"     : 0.3, 
	            "ridgeReg"          : 0.01, 
	            "mute"              : False 
			},
			'trained_model_filename'  : 'QGds02di02dm02p3.1000.1p8.ESN',
		},
		'TEST' : { 
			'testing_data_filename'   : 'QGds02di02dm02p3.TEST',
			'testing_length'		  : 3000,
			'trained_model_filename'  : 'QGds02di02dm02p3.1000.1p8.ESN', 
			'output_filenames'		  : {
				'stream_function_estimated' : 'QGds02di02dm02p3.1000.1p8.est',
				'stream_function_actual'    : 'QGds02di02dm02p3.1000.1p8.actual',
			}

		}
	},

	'QGds02di02dm02p3.1250.1p8': {
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
	            "resSize"           : 1250, 
	            "partial_know"      : False, 
	            "noise"             : 1e-2, 
	            "density"           : 1e-2, 
	            "spectral_radius"   : 1.8, 
	            "leaking_rate"      : 0.2, 
	            "input_scaling"     : 0.3, 
	            "ridgeReg"          : 0.01, 
	            "mute"              : False 
			},
			'trained_model_filename'  : 'QGds02di02dm02p3.1250.1p8.ESN',
		},
		'TEST' : { 
			'testing_data_filename'   : 'QGds02di02dm02p3.TEST',
			'testing_length'		  : 3000,
			'trained_model_filename'  : 'QGds02di02dm02p3.1250.1p8.ESN', 
			'output_filenames'		  : {
				'stream_function_estimated' : 'QGds02di02dm02p3.1250.1p8.est',
				'stream_function_actual'    : 'QGds02di02dm02p3.1250.1p8.actual',
			}

		}
	},

	'QGds02di02dm02p3.2000.1p8': {
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
	            "resSize"           : 2000, 
	            "partial_know"      : False, 
	            "noise"             : 1e-2, 
	            "density"           : 1e-2, 
	            "spectral_radius"   : 1.8, 
	            "leaking_rate"      : 0.2, 
	            "input_scaling"     : 0.3, 
	            "ridgeReg"          : 0.01, 
	            "mute"              : False 
			},
			'trained_model_filename'  : 'QGds02di02dm02p3.2000.1p8.ESN',
		},
		'TEST' : { 
			'testing_data_filename'   : 'QGds02di02dm02p3.TEST',
			'testing_length'		  : 3000,
			'trained_model_filename'  : 'QGds02di02dm02p3.2000.1p8.ESN', 
			'output_filenames'		  : {
				'stream_function_estimated' : 'QGds02di02dm02p3.2000.1p8.est',
				'stream_function_actual'    : 'QGds02di02dm02p3.2000.1p8.actual',
			}

		}
	},

	'QGds02di02dm02p3.100.1p8': {
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
	            "resSize"           : 100, 
	            "partial_know"      : False, 
	            "noise"             : 1e-2, 
	            "density"           : 1e-2, 
	            "spectral_radius"   : 1.8, 
	            "leaking_rate"      : 0.2, 
	            "input_scaling"     : 0.3, 
	            "ridgeReg"          : 0.01, 
	            "mute"              : False 
			},
			'trained_model_filename'  : 'QGds02di02dm02p3.100.1p8.ESN',
		},
		'TEST' : { 
			'testing_data_filename'   : 'QGds02di02dm02p3.TEST',
			'testing_length'		  : 3000,
			'trained_model_filename'  : 'QGds02di02dm02p3.100.1p8.ESN', 
			'output_filenames'		  : {
				'stream_function_estimated' : 'QGds02di02dm02p3.100.1p8.est',
				'stream_function_actual'    : 'QGds02di02dm02p3.100.1p8.actual',
			}

		}
	},

	'QGds02di02dm02p3.10000.1p8': {
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
	            "resSize"           : 10000, 
	            "partial_know"      : False, 
	            "noise"             : 1e-2, 
	            "density"           : 1e-1, 
	            "spectral_radius"   : 1.8, 
	            "leaking_rate"      : 0.2, 
	            "input_scaling"     : 0.3, 
	            "ridgeReg"          : 0.01, 
	            "mute"              : False 
			},
			'trained_model_filename'  : 'QGds02di02dm02p3.10000.1p8.ESN',
		},
		'TEST' : { 
			'testing_data_filename'   : 'QGds02di02dm02p3.TEST',
			'testing_length'		  : 3000,
			'trained_model_filename'  : 'QGds02di02dm02p3.10000.1p8.ESN', 
			'output_filenames'		  : {
				'stream_function_estimated' : 'QGds02di02dm02p3.10000.1p8.est',
				'stream_function_actual'    : 'QGds02di02dm02p3.10000.1p8.actual',
			}

		}
	},


######################################################################################
# testing spectral radius 

	'QGds02di02dm02p3.1000.0p3': {
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
	            "density"           : 1e-1, 
	            "spectral_radius"   : 0.3, 
	            "leaking_rate"      : 0.2, 
	            "input_scaling"     : 0.3, 
	            "ridgeReg"          : 0.01, 
	            "mute"              : False 
			},
			'trained_model_filename'  : 'QGds02di02dm02p3.1000.0p3.ESN',
		},
		'TEST' : { 
			'testing_data_filename'   : 'QGds02di02dm02p3.TEST',
			'testing_length'		  : 3000,
			'trained_model_filename'  : 'QGds02di02dm02p3.1000.0p3.ESN', 
			'output_filenames'		  : { # Davis wuz here!!
				'stream_function_estimated' : 'QGds02di02dm02p3.1000.0p3.est',
				'stream_function_actual'    : 'QGds02di02dm02p3.1000.0p3.actual',
			}

		}
	},

##################################################Davis#########################################

	'DOUBLE_GYRE_DEMO': {		
		'PREPROCESS' : { 
			'sf_filename' : 'dgsf_0p01_200_128_64_0p1_0p2', 
			'training_length'		  : 10000,
			'training_data_filename'  : 'dgsf_0p01_200_128_64_0p1_0p2.TRAIN',
			'testing_data_filename'  : 'dgsf_0p01_200_128_64_0p1_0p2.TEST'	
		},
		'TRAIN' : { 
			'training_data_filename'  : 'dgsf_0p01_200_128_64_0p1_0p2.TRAIN',
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
			'trained_model_filename'  : 'dgsf_0p01_200_128_64_0p1_0p2.ESN',
		},
		'TEST' : { 
			'testing_data_filename'   : 'dgsf_0p01_200_128_64_0p1_0p2.TEST',
			'testing_length'		  : 5000,
			'trained_model_filename'  : 'dgsf_0p01_200_128_64_0p1_0p2.ESN', 
			'output_filenames'		  : {
				'stream_function_estimated' : 'dgsf_0p01_200_128_64_0p1_0p2.est',
				'stream_function_actual'    : 'dgsf_0p01_200_128_64_0p1_0p2.actual',
			}

		}
	}, # Davis wuz here!!



}