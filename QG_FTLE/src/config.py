INPUT_PATH_DIR = "../inputs"
# DATA_PATH_DIR = "D:/Documents/Thesis/QG_FTLE/data"
DISCRETE_VELOCITY_PATH_DIR = "../data/discrete_velocity_fields"
INTERP_VELOCITY_PATH_DIR = "../data/interp_velocity_fields"
FTLE_MAPPING_PATH_DIR = "../data/FTLEmapping"
FTLE_FIELDS_PATH_DIR = "../data/FTLEfields"
RESULT_PATH_DIR = "../results"


_TEST_XCT = 80
_TEST_YCT = 40


CONFIGS = { 
    'TESTING' : {
        'GENERATE_STREAM_FUNCTION_FIELDS' : {
            'dt' : 0.1, 
            'elapsedTime' : 20, 
            'xct': _TEST_XCT, 
            'yct': _TEST_YCT,
            'stream_function_filename' : 'test.dgsf.{0}.{1}.actual'.format(_TEST_XCT, _TEST_YCT)
        }, 
        'GENERATE_VELOCITY_FIELDS' : {
            'stream_function_filename' : 'test.dgsf.{0}.{1}.actual'.format(_TEST_XCT, _TEST_YCT), 
            'velocity_filename' : 'test.dguv.{0}.{1}.actual'.format(_TEST_XCT, _TEST_YCT),
            'velocity_func_filename' : 'test.dguv.{0}.{1}.actual.interp'.format(_TEST_XCT, _TEST_YCT)
        }, 
        'GENERATE_FTLE_MAPPING' : {
            'iters' : 10, 
            'mapped_dt' : 10,
            'dt' : 0.1,
            'xct': _TEST_XCT, 
            'yct': _TEST_YCT,
            'velocity_func_filename': 'test.dguv.{0}.{1}.actual.interp'.format(_TEST_XCT, _TEST_YCT),
            'mapping_path_dir': 'test.dgsf.{0}.{1}.actual'.format(_TEST_XCT, _TEST_YCT)
        },
        'GENERATE_FTLE_FIELDS': {
            'iters' : 10,
            'xct': _TEST_XCT, 
            'yct': _TEST_YCT,
            'mapping_path_dir': 'test.dgsf.{0}.{1}.actual'.format(_TEST_XCT, _TEST_YCT),
            'ftle_path_dir' : 'test.dgsf.{0}.{1}.actual'.format(_TEST_XCT, _TEST_YCT)
        }, 
        'GENERATE_FTLE_ANIMATIONS': {
            'iters' : 10,
            'xct': _TEST_XCT, 
            'yct': _TEST_YCT,
            'ftle_path_dir' : 'test.dgsf.{0}.{1}.actual'.format(_TEST_XCT, _TEST_YCT),
            'ftle_animation_filename' : 'test.dgsf.{0}.{1}.actual.gif'.format(_TEST_XCT, _TEST_YCT),
        }  
    },

    'DOUBLE_GYRE_ACTUAL' : {
        'GENERATE_STREAM_FUNCTION_FIELDS' : {
            'dt' : 0.1, 
            'elapsedTime' : 20, 
            'xct': 128, 
            'yct': 64,
            'stream_function_filename' : 'dgsf.{0}.{1}.actual'.format(128, 64)
        }, 
        'GENERATE_VELOCITY_FIELDS' : {
            'stream_function_filename' : 'dgsf.{0}.{1}.actual'.format(128, 64), 
            'velocity_filename' : 'dguv.{0}.{1}.actual'.format(128, 64),
            'velocity_func_filename' : 'dguv.{0}.{1}.actual.interp'.format(128, 64)
        }, 
        'GENERATE_FTLE_MAPPING' : {
            'iters' : 10, 
            'mapped_dt' : 10,
            'dt' : 0.1,
            'xct': 128, 
            'yct': 64,
            'velocity_func_filename': 'dguv.{0}.{1}.actual.interp'.format(128, 64),
            'mapping_path_dir': 'dgsf.{0}.{1}.actual'.format(128, 64)
        },
        'GENERATE_FTLE_FIELDS': {
            'iters' : 10,
            'xct': 128, 
            'yct': 64,
            'mapping_path_dir': 'dgsf.{0}.{1}.actual'.format(128, 64),
            'ftle_path_dir' : 'dgsf.{0}.{1}.actual'.format(128, 64)
        }, 
        'GENERATE_FTLE_ANIMATIONS': {
            'iters' : 10,
            'xct': 128, 
            'yct': 64,
            'ftle_path_dir' : 'dgsf.{0}.{1}.actual'.format(128, 64),
            'ftle_animation_filename' : 'dgsf.{0}.{1}.actual.gif'.format(128, 64),
        }  
    },

    'DOUBLE_GYRE_EST' : { 
        'GENERATE_VELOCITY_FIELDS' : {
            'stream_function_filename' : 'dgsf.{0}.{1}.est'.format(128, 64), 
            'velocity_filename' : 'dguv.{0}.{1}.est'.format(128, 64),
            'velocity_func_filename' : 'dguv.{0}.{1}.est.interp'.format(128, 64)
        }, 
        'GENERATE_FTLE_MAPPING' : {
            'iters' : 10, 
            'mapped_dt' : 10,
            'dt' : 0.1,
            'xct': 128, 
            'yct': 64,
            'velocity_func_filename': 'dguv.{0}.{1}.est.interp'.format(128, 64),
            'mapping_path_dir': 'dgsf.{0}.{1}.est'.format(128, 64)
        },
        'GENERATE_FTLE_FIELDS': {
            'iters' : 10,
            'xct': 128, 
            'yct': 64,
            'mapping_path_dir': 'dgsf.{0}.{1}.est'.format(128, 64),
            'ftle_path_dir' : 'dgsf.{0}.{1}.est'.format(128, 64)
        }, 
        'GENERATE_FTLE_ANIMATIONS': {
            'iters' : 10,
            'xct': 128, 
            'yct': 64,
            'ftle_path_dir' : 'dgsf.{0}.{1}.est'.format(128, 64),
            'ftle_animation_filename' : 'dgsf.{0}.{1}.est.gif'.format(128, 64),
        }  
    },

    'QUASIGEO_ACTUAL' : { 
        'GENERATE_VELOCITY_FIELDS' : {
            'stream_function_filename' : 'qgsf.{0}.{1}.actual'.format(64, 128), 
            'velocity_filename' : 'qguv.{0}.{1}.actual'.format(64, 128),
            'velocity_func_filename' : 'qguv.{0}.{1}.actual.interp'.format(64, 128)
        }, 
        'GENERATE_FTLE_MAPPING' : {
            'iters' : 20, 
            'mapped_dt' : 2,
            'dt' : 0.01,
            'xct': 64, 
            'yct': 128,
            'velocity_func_filename': 'qguv.{0}.{1}.actual.interp'.format(64, 128),
            'mapping_path_dir': 'qgsf.{0}.{1}.actual'.format(64, 128)
        },
        'GENERATE_FTLE_FIELDS': {
            'iters' : 20,
            'xct': 64, 
            'yct': 128,
            'mapping_path_dir': 'qgsf.{0}.{1}.actual'.format(64, 128),
            'ftle_path_dir' : 'qgsf.{0}.{1}.actual'.format(64, 128)
        }, 
        'GENERATE_FTLE_ANIMATIONS': {
            'iters' : 20,
            'xct': 64, 
            'yct': 128,
            'ftle_path_dir' : 'qgsf.{0}.{1}.actual'.format(64, 128),
            'ftle_animation_filename' : 'qgsf.{0}.{1}.actual.gif'.format(64, 128),
        }
    },

    'QUASIGEO_EST' : { 
        'GENERATE_VELOCITY_FIELDS' : {
            'stream_function_filename' : 'qgsf.{0}.{1}.est'.format(64, 128), 
            'velocity_filename' : 'qguv.{0}.{1}.est'.format(64, 128),
            'velocity_func_filename' : 'qguv.{0}.{1}.est.interp'.format(64, 128)
        }, 
        'GENERATE_FTLE_MAPPING' : {
            'iters' : 20, 
            'mapped_dt' : 2,
            'dt' : 0.01,
            'xct': 64, 
            'yct': 128,
            'velocity_func_filename': 'qguv.{0}.{1}.est.interp'.format(64, 128),
            'mapping_path_dir': 'qgsf.{0}.{1}.est'.format(64, 128)
        },
        'GENERATE_FTLE_FIELDS': {
            'iters' : 20,
            'xct': 64, 
            'yct': 128,
            'mapping_path_dir': 'qgsf.{0}.{1}.est'.format(64, 128),
            'ftle_path_dir' : 'qgsf.{0}.{1}.est'.format(64, 128)
        }, 
        'GENERATE_FTLE_ANIMATIONS': {
            'iters' : 20,
            'xct': 64, 
            'yct': 128,
            'ftle_path_dir' : 'qgsf.{0}.{1}.est'.format(64, 128),
            'ftle_animation_filename' : 'qgsf.{0}.{1}.est.gif'.format(64, 128),
        }  
    }


}