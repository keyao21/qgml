import pickle 
import os 
import sys 
import shutil 

curr_fullpath = os.getcwd()
ML_Fluid_fullpath = os.path.abspath("../ML_Fluid/src/")
QG_FTLE_fullpath = os.path.abspath("../QG_FTLE/src/")
ML_Fluid_RESULTS_fullpath = os.path.abspath("../ML_Fluid/results/")
QG_FTLE_INPUTS_fullpath = os.path.abspath("../QG_FTLE/inputs/")

"""
Some notes regarding functions switch_to_qgftle_src_dir() and switch_to_mlfluids_src_dir(): 

It is necessary to change the directory to import the python modules in the relevant directory
(e.g. QG_FTLE or ML_Fluids); however, it is **ALSO** necessary to insert the src to the top of the
path -- this is because there may be modules which share names (e.g. configs.py) across both QG_FTLE
and ML_Fluids src directories, so the namespace must be explicit and the correct src dir must be 
at the top of the path with highest precedence. We must delete the modules that share names

"""
def switch_to_qgftle_src_dir(): 
	os.chdir(QG_FTLE_fullpath)
	sys.path.insert(0,QG_FTLE_fullpath)
	# delete overlapping modules with same name
	for module_name in ['config', 'util']: 
		try:
			del sys.modules[module_name]
		except: 
			pass 

def switch_to_home_dir(): 
	os.chdir(curr_fullpath)

def switch_to_mlfluids_src_dir(): 
	os.chdir(ML_Fluid_fullpath)
	# delete overlapping modules with same name
	sys.path.insert(0,ML_Fluid_fullpath)
	# delete overlapping modules with same name
	for module_name in ['config', 'util']: 
		try:
			del sys.modules[module_name]
		except: 
			pass 

def generate_experiment_id(): 
	# from experiment parmas create unique string to be used as key for config dicts 
	pass

def generate_ml_fluid_params(  training_length, trained_model_params, testing_length, trained_model_filename, stream_function_prefix ): 
	# params to pass into config for ML_Fluid 
	params_dict = {
		'PREPROCESS' : { 
			'MATLAB_filename' : 'QG_psi_ds0.02_di0.02_dm0.02_pertamp0.3.mat', 
			'training_length'		  : training_length,
			'training_data_filename'  : 'QGds02di02dm02p3.TRAIN',
			'testing_data_filename'  : 'QGds02di02dm02p3.TEST'	
		},
		'TRAIN' : { 
			'training_data_filename'  : 'QGds02di02dm02p3.TRAIN',
			'trained_model_params'    : trained_model_params,
			'trained_model_filename'  : trained_model_filename,
		},
		'TEST' : { 
			'testing_data_filename'   : 'QGds02di02dm02p3.TEST',
			'testing_length'		  : testing_length,
			'trained_model_filename'  : trained_model_filename, 
			'output_filenames'		  : { # Davis wuz here!!
				'stream_function_estimated' : f'{stream_function_prefix}.est',
				'stream_function_actual'    : f'{stream_function_prefix}.actual',
			}

		}
	}
	return params_dict

def generate_qgftle_params( stream_function_prefix ):
	"""
	params to pass into config for QG_FTLE
	stream_function_prefix - part of the stream function file name 
	**BEFORE** est and actual, e.g. QGds02di02dm02p3.1000.0p3

	return a dict with keys: <stream_function_prefix>.actual and <stream_function_prefix>.est
	"""

	params_dict = dict()
	for actual_flag in ['actual', 'est']: 	
		params_dict[ f"{stream_function_prefix}.{actual_flag}" ] = { 
	        'GENERATE_VELOCITY_FIELDS' : {
	            'stream_function_filename' : f"{stream_function_prefix}.{actual_flag}", 
	            'velocity_filename' : f"{stream_function_prefix}.uv.{actual_flag}",
	            'velocity_func_filename' : f"{stream_function_prefix}.uvinterp.{actual_flag}"
	        }, 
	        'GENERATE_FTLE_MAPPING' : {
	            'iters' : 10, 
	            'mapped_dt' : 3,
	            'dt' : 0.01,
	            'xct': 64, 
	            'yct': 128,
	            # Davis wuz here!!
	            'velocity_func_filename': f"{stream_function_prefix}.uvinterp.{actual_flag}",
	            'mapping_path_dir': f"{stream_function_prefix}.{actual_flag}"
	        },
	        'GENERATE_FTLE_FIELDS': {
	            'iters' : 10,
	            'xct': 64, 
	            'yct': 128,
	            'mapping_path_dir': f"{stream_function_prefix}.{actual_flag}",
	            'ftle_path_dir' : f"{stream_function_prefix}.{actual_flag}"
	        }, 
	        'GENERATE_FTLE_ANIMATIONS': {
	            'iters' : 10,
	            'xct': 64, 
	            'yct': 128,
	            'ftle_path_dir' : f"{stream_function_prefix}.{actual_flag}",
	            'ftle_animation_filename' : f"{stream_function_prefix}.{actual_flag}.gif",
	        }
	    }
	return params_dict

def run_experiment(resSize, spectral_radius): 
	## Preparing parameters for entire experiment
	## ml fluid params should link to qgftle params by  
	## stream_function_estimated and stream_function_actual
	trained_model_params = { 
	    "initLen"           : 0, 
	    "resSize"           : resSize, 
	    "partial_know"      : False, 
	    "noise"             : 1e-2, 
	    "density"           : 1e-1, 
	    "spectral_radius"   : spectral_radius, 
	    "leaking_rate"      : 0.2, 
	    "input_scaling"     : 0.3, 
	    "ridgeReg"          : 0.01, 
	    "mute"              : False 
	}
	training_length = 1000
	testing_length = 3000
	stream_function_prefix = f"QGds02di02dm02p3_{resSize}_{spectral_radius:.1f}"
	trained_model_filename = f'{stream_function_prefix}.ESN'
	ml_fluid_params_dict = generate_ml_fluid_params(	training_length=training_length, 
										trained_model_params=trained_model_params,
										testing_length=testing_length,
										trained_model_filename=trained_model_filename,
										stream_function_prefix=stream_function_prefix
									)
	qgftle_params_dict = generate_qgftle_params( stream_function_prefix=stream_function_prefix )

	"""
	1. Running ML_Fluids procedure - train a model on training data and generate 
	and testing sample to compare. Specifically, we must
		a. Preprocess data 
		b. train data 
		c. test data 
	"""
	# ensure correct directory
	switch_to_mlfluids_src_dir()
	print(os.getcwd())
	# a. preprocessing data
	import preprocess
	preprocess.preprocess_input_data( **ml_fluid_params_dict['PREPROCESS'] )
	# b. training data
	import train 
	train.train_ESN(**ml_fluid_params_dict['TRAIN'])
	# c. testing data 
	import test 
	test.test_ESN(**ml_fluid_params_dict['TEST'])
	switch_to_home_dir()

	## COPY STREAM FUNCTION FILES OVER TO QG_FTLE INPUT DIRECTORY
	MLFluid_streamfunction_fullpaths = [os.path.join(ML_Fluid_RESULTS_fullpath,streamfunction_filename) 
						for _,streamfunction_filename in ml_fluid_params_dict['TEST']['output_filenames'].items()]
	for streamfunction_fullpath in MLFluid_streamfunction_fullpaths:
		# qgftle_streamfunction_fullpath = os.path.join(QG_FTLE_INPUTS_fullpath, os.path.basename(streamfunction_fullpath))
		shutil.copy(streamfunction_fullpath, QG_FTLE_INPUTS_fullpath)


	"""
	2. Running QG_FTLE procedure **FOR BOTH estimated and actual stream function files
	- interpolate discrete velocity fields to continuous, using them to generate ftle mappings and fields
		a. generate velcoity fields 
		b. generate ftle mappings 
		c. generate ftle fields 
	"""
	# ensure correct directory
	for params_key, params_dict in qgftle_params_dict.items():
		# ensure correct directory
		switch_to_qgftle_src_dir()
		# a. generate velcoity fields
		import generate_velocity_fields
		generate_velocity_fields.generate_velocity_fields( **params_dict['GENERATE_VELOCITY_FIELDS'] )
		# b. generate ftle mappings
		import generate_FTLE_mapping
		generate_FTLE_mapping.generate_mapping_files( **params_dict['GENERATE_FTLE_MAPPING'] )
		# c. gerenate ftle files
		import generate_FTLE_fields 
		generate_FTLE_fields.generate_FTLE_fields( **params_dict['GENERATE_FTLE_FIELDS'] )
	# d. compare ftle files 	
	# ensure correct directory
	switch_to_qgftle_src_dir()
	import compare_FTLE_fields 
	ssi = compare_FTLE_fields.compare_FTLE_animation(iters=10, ftle_path_dirs=[ params_dict['GENERATE_FTLE_FIELDS']['ftle_path_dir'] 
																			for _ , params_dict in qgftle_params_dict.items() ],
														ftle_animation_filename=f"{stream_function_prefix}.gif")
	switch_to_home_dir()
	return ssi 

if __name__ == '__main__':
	

	# resSize = 1000
	# spectral_radius = 0.3

	results = []
	for spectral_radius in [float(i*3/10.0) for i in range(1,11)]:
		for resSize in [100,1000,10000]: 
			try: # wrap in try except block in case of memory issues...
				ssi = run_experiment(resSize=resSize, spectral_radius=spectral_radius)
				results.append([ssi, resSize, spectral_radius])
			except MemoryError:
				print('memory error...skipping')

	import pandas as pd 
	df = pd.DataFrame(results,columns=['SSI','resSize','spectral_density'])  
	df = df.pivot_table(values='SSI',index='spectral_density',columns='resSize')
	df.to_clipboard()
	df.to_csv('experiment_results.csv')
	import pdb;pdb.set_trace()




