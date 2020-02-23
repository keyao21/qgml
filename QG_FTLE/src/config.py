INPUT_PATH_DIR = "../inputs"
# DATA_PATH_DIR = "D:/Documents/Thesis/QG_FTLE/data"
DISCRETE_VELOCITY_PATH_DIR = "../data/discrete_velocity_fields"
INTERP_VELOCITY_PATH_DIR = "../data/interp_velocity_fields"
FTLE_MAPPING_PATH_DIR = "../data/FTLEmapping"
FTLE_FIELDS_PATH_DIR = "../data/FTLEfields"
RESULT_PATH_DIR = "../results"


_TEST_XCT = 20
_TEST_YCT = 10


CONFIGS = { 
    'TESTING' : {
        'GENERATE_STREAM_FUNCTION_FIELDS' : {
            'dt' : 0.1, 
            'elapsedTime' : 20, 
            'xct': _TEST_XCT, 
            'yct': _TEST_YCT,
            'amp': 0.2,
            'epsilon': 0.3,
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
        } , 
    },

    'DOUBLE_GYRE_ACTUAL' : {
        'GENERATE_STREAM_FUNCTION_FIELDS' : {
            'dt' : 0.1, 
            'elapsedTime' : 20, 
            'xct': 128, 
            'yct': 64,
            'amp': 0.2,
            'epsilon': 0.3,
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
    },


##########################################################################################
# quasigeo experiment configs to test ssi for changing ML params (res size and spec rad)
    'QGds02di02dm02p3.1000.1p8.actual' : { 
        'GENERATE_VELOCITY_FIELDS' : {
            'stream_function_filename' : 'QGds02di02dm02p3.1000.1p8.actual', 
            'velocity_filename' : 'QGds02di02dm02p3.1000.1p8.uv.actual',
            'velocity_func_filename' : 'QGds02di02dm02p3.1000.1p8.uvinterp.actual'
        }, 
        'GENERATE_FTLE_MAPPING' : {
            'iters' : 20, 
            'mapped_dt' : 3,
            'dt' : 0.01,
            'xct': 64, 
            'yct': 128,
            'velocity_func_filename': 'QGds02di02dm02p3.1000.1p8.uvinterp.actual',
            'mapping_path_dir': 'QGds02di02dm02p3.1000.1p8.actual'
        },
        'GENERATE_FTLE_FIELDS': {
            'iters' : 20,
            'xct': 64, 
            'yct': 128,
            'mapping_path_dir': 'QGds02di02dm02p3.1000.1p8.actual',
            'ftle_path_dir' : 'QGds02di02dm02p3.1000.1p8.actual'
        }, 
        'GENERATE_FTLE_ANIMATIONS': {
            'iters' : 20,
            'xct': 64, 
            'yct': 128,
            'ftle_path_dir' : 'QGds02di02dm02p3.1000.1p8.actual',
            'ftle_animation_filename' : 'QGds02di02dm02p3.1000.1p8.actual.gif',
        }
    },

    'QGds02di02dm02p3.1000.1p8.est' : { 
        'GENERATE_VELOCITY_FIELDS' : {
            'stream_function_filename' : 'QGds02di02dm02p3.1000.1p8.est', 
            'velocity_filename' : 'QGds02di02dm02p3.1000.1p8.uv.est',
            'velocity_func_filename' : 'QGds02di02dm02p3.1000.1p8.uvinterp.est'
        }, 
        'GENERATE_FTLE_MAPPING' : {
            'iters' : 20, 
            'mapped_dt' : 3,
            'dt' : 0.01,
            'xct': 64, 
            'yct': 128,
            'velocity_func_filename': 'QGds02di02dm02p3.1000.1p8.uvinterp.est',
            'mapping_path_dir': 'QGds02di02dm02p3.1000.1p8.est'
        },
        'GENERATE_FTLE_FIELDS': {
            'iters' : 20,
            'xct': 64, 
            'yct': 128,
            'mapping_path_dir': 'QGds02di02dm02p3.1000.1p8.est',
            'ftle_path_dir' : 'QGds02di02dm02p3.1000.1p8.est'
        }, 
        'GENERATE_FTLE_ANIMATIONS': {
            'iters' : 20,
            'xct': 64, 
            'yct': 128,
            'ftle_path_dir' : 'QGds02di02dm02p3.1000.1p8.est',
            'ftle_animation_filename' : 'QGds02di02dm02p3.1000.1p8.est.gif',
        }
    },


    'QGds02di02dm02p3.1250.1p8.actual' : { 
        'GENERATE_VELOCITY_FIELDS' : {
            'stream_function_filename' : 'QGds02di02dm02p3.1250.1p8.actual', 
            'velocity_filename' : 'QGds02di02dm02p3.1250.1p8.uv.actual',
            'velocity_func_filename' : 'QGds02di02dm02p3.1250.1p8.uvinterp.actual'
        }, 
        'GENERATE_FTLE_MAPPING' : {
            'iters' : 20, 
            'mapped_dt' : 3,
            'dt' : 0.01,
            'xct': 64, 
            'yct': 128,
            'velocity_func_filename': 'QGds02di02dm02p3.1250.1p8.uvinterp.actual',
            'mapping_path_dir': 'QGds02di02dm02p3.1250.1p8.actual'
        },
        'GENERATE_FTLE_FIELDS': {
            'iters' : 20,
            'xct': 64, 
            'yct': 128,
            'mapping_path_dir': 'QGds02di02dm02p3.1250.1p8.actual',
            'ftle_path_dir' : 'QGds02di02dm02p3.1250.1p8.actual'
        }, 
        'GENERATE_FTLE_ANIMATIONS': {
            'iters' : 20,
            'xct': 64, 
            'yct': 128,
            'ftle_path_dir' : 'QGds02di02dm02p3.1250.1p8.actual',
            'ftle_animation_filename' : 'QGds02di02dm02p3.1250.1p8.actual.gif',
        }
    },

    'QGds02di02dm02p3.1250.1p8.est' : { 
        'GENERATE_VELOCITY_FIELDS' : {
            'stream_function_filename' : 'QGds02di02dm02p3.1250.1p8.est', 
            'velocity_filename' : 'QGds02di02dm02p3.1250.1p8.uv.est',
            'velocity_func_filename' : 'QGds02di02dm02p3.1250.1p8.uvinterp.est'
        }, 
        'GENERATE_FTLE_MAPPING' : {
            'iters' : 20, 
            'mapped_dt' : 3,
            'dt' : 0.01,
            'xct': 64, 
            'yct': 128,
            'velocity_func_filename': 'QGds02di02dm02p3.1250.1p8.uvinterp.est',
            'mapping_path_dir': 'QGds02di02dm02p3.1250.1p8.est'
        },
        'GENERATE_FTLE_FIELDS': {
            'iters' : 20,
            'xct': 64, 
            'yct': 128,
            'mapping_path_dir': 'QGds02di02dm02p3.1250.1p8.est',
            'ftle_path_dir' : 'QGds02di02dm02p3.1250.1p8.est'
        }, 
        'GENERATE_FTLE_ANIMATIONS': {
            'iters' : 20,
            'xct': 64, 
            'yct': 128,
            'ftle_path_dir' : 'QGds02di02dm02p3.1250.1p8.est',
            'ftle_animation_filename' : 'QGds02di02dm02p3.1250.1p8.est.gif',
        }
    },


    'QGds02di02dm02p3.2000.1p8.actual' : { 
        'GENERATE_VELOCITY_FIELDS' : {
            'stream_function_filename' : 'QGds02di02dm02p3.2000.1p8.actual', 
            'velocity_filename' : 'QGds02di02dm02p3.2000.1p8.uv.actual',
            'velocity_func_filename' : 'QGds02di02dm02p3.2000.1p8.uvinterp.actual'
        }, 
        'GENERATE_FTLE_MAPPING' : {
            'iters' : 20, 
            'mapped_dt' : 3,
            'dt' : 0.01,
            'xct': 64, 
            'yct': 128,
            'velocity_func_filename': 'QGds02di02dm02p3.2000.1p8.uvinterp.actual',
            'mapping_path_dir': 'QGds02di02dm02p3.2000.1p8.actual'
        },
        'GENERATE_FTLE_FIELDS': {
            'iters' : 20,
            'xct': 64, 
            'yct': 128,
            'mapping_path_dir': 'QGds02di02dm02p3.2000.1p8.actual',
            'ftle_path_dir' : 'QGds02di02dm02p3.2000.1p8.actual'
        }, 
        'GENERATE_FTLE_ANIMATIONS': {
            'iters' : 20,
            'xct': 64, 
            'yct': 128,
            'ftle_path_dir' : 'QGds02di02dm02p3.2000.1p8.actual',
            'ftle_animation_filename' : 'QGds02di02dm02p3.2000.1p8.actual.gif',
        }
    },

    'QGds02di02dm02p3.2000.1p8.est' : { 
        'GENERATE_VELOCITY_FIELDS' : {
            'stream_function_filename' : 'QGds02di02dm02p3.2000.1p8.est', 
            'velocity_filename' : 'QGds02di02dm02p3.2000.1p8.uv.est',
            'velocity_func_filename' : 'QGds02di02dm02p3.2000.1p8.uvinterp.est'
        }, 
        'GENERATE_FTLE_MAPPING' : {
            'iters' : 20, 
            'mapped_dt' : 3,
            'dt' : 0.01,
            'xct': 64, 
            'yct': 128,
            'velocity_func_filename': 'QGds02di02dm02p3.2000.1p8.uvinterp.est',
            'mapping_path_dir': 'QGds02di02dm02p3.2000.1p8.est'
        },
        'GENERATE_FTLE_FIELDS': {
            'iters' : 20,
            'xct': 64, 
            'yct': 128,
            'mapping_path_dir': 'QGds02di02dm02p3.2000.1p8.est',
            'ftle_path_dir' : 'QGds02di02dm02p3.2000.1p8.est'
        }, 
        'GENERATE_FTLE_ANIMATIONS': {
            'iters' : 20,
            'xct': 64, 
            'yct': 128,
            'ftle_path_dir' : 'QGds02di02dm02p3.2000.1p8.est',
            'ftle_animation_filename' : 'QGds02di02dm02p3.2000.1p8.est.gif',
        }
    },

    'QGds02di02dm02p3.100.1p8.actual' : { 
        'GENERATE_VELOCITY_FIELDS' : {
            'stream_function_filename' : 'QGds02di02dm02p3.100.1p8.actual', 
            'velocity_filename' : 'QGds02di02dm02p3.100.1p8.uv.actual',
            'velocity_func_filename' : 'QGds02di02dm02p3.100.1p8.uvinterp.actual'
        }, 
        'GENERATE_FTLE_MAPPING' : {
            'iters' : 20, 
            'mapped_dt' : 3,
            'dt' : 0.01,
            'xct': 64, 
            'yct': 128,
            'velocity_func_filename': 'QGds02di02dm02p3.100.1p8.uvinterp.actual',
            'mapping_path_dir': 'QGds02di02dm02p3.100.1p8.actual'
        },
        'GENERATE_FTLE_FIELDS': {
            'iters' : 20,
            'xct': 64, 
            'yct': 128,
            'mapping_path_dir': 'QGds02di02dm02p3.100.1p8.actual',
            'ftle_path_dir' : 'QGds02di02dm02p3.100.1p8.actual'
        }, 
        'GENERATE_FTLE_ANIMATIONS': {
            'iters' : 20,
            'xct': 64, 
            'yct': 128,
            'ftle_path_dir' : 'QGds02di02dm02p3.100.1p8.actual',
            'ftle_animation_filename' : 'QGds02di02dm02p3.100.1p8.actual.gif',
        }
    },

    'QGds02di02dm02p3.100.1p8.est' : { 
        'GENERATE_VELOCITY_FIELDS' : {
            'stream_function_filename' : 'QGds02di02dm02p3.100.1p8.est', 
            'velocity_filename' : 'QGds02di02dm02p3.100.1p8.uv.est',
            'velocity_func_filename' : 'QGds02di02dm02p3.100.1p8.uvinterp.est'
        }, 
        'GENERATE_FTLE_MAPPING' : {
            'iters' : 20, 
            'mapped_dt' : 3,
            'dt' : 0.01,
            'xct': 64, 
            'yct': 128,
            'velocity_func_filename': 'QGds02di02dm02p3.100.1p8.uvinterp.est',
            'mapping_path_dir': 'QGds02di02dm02p3.100.1p8.est'
        },
        'GENERATE_FTLE_FIELDS': {
            'iters' : 20,
            'xct': 64, 
            'yct': 128,
            'mapping_path_dir': 'QGds02di02dm02p3.100.1p8.est',
            'ftle_path_dir' : 'QGds02di02dm02p3.100.1p8.est'
        }, 
        'GENERATE_FTLE_ANIMATIONS': {
            'iters' : 20,
            'xct': 64, 
            'yct': 128,
            'ftle_path_dir' : 'QGds02di02dm02p3.100.1p8.est',
            'ftle_animation_filename' : 'QGds02di02dm02p3.100.1p8.est.gif',
        }
    },

###########################################################################################


    'dgsf_0p01_200_128_64_0p1_0p2.actual' : { 
        'GENERATE_VELOCITY_FIELDS' : {
            'stream_function_filename' : 'dgsf_0p01_200_128_64_0p1_0p2.actual', 
            'velocity_filename' : 'dgsf_0p01_200_128_64_0p1_0p2.uv.actual',
            'velocity_func_filename' : 'dgsf_0p01_200_128_64_0p1_0p2.uvinterp.actual'
        }, 
        'GENERATE_FTLE_MAPPING' : {
            'iters' : 20, 
            'mapped_dt' : 10,
            'dt' : 0.01,
            'xct': 128, 
            'yct': 64,
            'velocity_func_filename': 'dgsf_0p01_200_128_64_0p1_0p2.uvinterp.actual',
            'mapping_path_dir': 'dgsf_0p01_200_128_64_0p1_0p2.actual'
        },
        'GENERATE_FTLE_FIELDS': {
            'iters' : 20,
            'xct': 128, 
            'yct': 64,
            'mapping_path_dir': 'dgsf_0p01_200_128_64_0p1_0p2.actual',
            'ftle_path_dir' : 'dgsf_0p01_200_128_64_0p1_0p2.actual'
        }, 
        'GENERATE_FTLE_ANIMATIONS': {
            'iters' : 20,
            'xct': 128, 
            'yct': 64,
            'ftle_path_dir' : 'dgsf_0p01_200_128_64_0p1_0p2.actual',
            'ftle_animation_filename' : 'dgsf_0p01_200_128_64_0p1_0p2.actual.gif',
        }
    },

    'dgsf_0p01_200_128_64_0p1_0p2.est' : { 
        'GENERATE_VELOCITY_FIELDS' : {
            'stream_function_filename' : 'dgsf_0p01_200_128_64_0p1_0p2.est', 
            'velocity_filename' : 'dgsf_0p01_200_128_64_0p1_0p2.uv.est',
            'velocity_func_filename' : 'dgsf_0p01_200_128_64_0p1_0p2.uvinterp.est'
        }, 
        'GENERATE_FTLE_MAPPING' : {
            'iters' : 20, 
            'mapped_dt' : 10,
            'dt' : 0.01,
            'xct': 128, 
            'yct': 64,
            'velocity_func_filename': 'dgsf_0p01_200_128_64_0p1_0p2.uvinterp.est',
            'mapping_path_dir': 'dgsf_0p01_200_128_64_0p1_0p2.est'
        },
        'GENERATE_FTLE_FIELDS': {
            'iters' : 20,
            'xct': 128, 
            'yct': 64,
            'mapping_path_dir': 'dgsf_0p01_200_128_64_0p1_0p2.est',
            'ftle_path_dir' : 'dgsf_0p01_200_128_64_0p1_0p2.est'
        }, 
        'GENERATE_FTLE_ANIMATIONS': {
            'iters' : 20,
            'xct': 128, 
            'yct': 64,
            'ftle_path_dir' : 'dgsf_0p01_200_128_64_0p1_0p2.est',
            'ftle_animation_filename' : 'dgsf_0p01_200_128_64_0p1_0p2.est.gif',
        }
    }
}