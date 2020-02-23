import unittest
import os 
import interp 
from config import * 
import numpy as np 
import random 
import util
import generate_velocity_fields, double_gyre

class Test_interp(unittest.TestCase):
    
    def setUp(self): 
        print('Setting up testing files...')
        TESTING_GENERATE_STREAM_FUNCTION_FIELDS_DICT = CONFIGS['TESTING']['GENERATE_STREAM_FUNCTION_FIELDS']
        self.xct, self.yct, self.dt = TESTING_GENERATE_STREAM_FUNCTION_FIELDS_DICT['xct'], \
                                    TESTING_GENERATE_STREAM_FUNCTION_FIELDS_DICT['yct'], \
                                    TESTING_GENERATE_STREAM_FUNCTION_FIELDS_DICT['dt']

        self.stream_function_filename = CONFIGS['TESTING']['GENERATE_STREAM_FUNCTION_FIELDS']['stream_function_filename']
        self.velocity_filename = CONFIGS['TESTING']['GENERATE_VELOCITY_FIELDS']['velocity_filename']
        self.velocity_func_filename = CONFIGS['TESTING']['GENERATE_VELOCITY_FIELDS']['velocity_func_filename']

        if not os.path.exists( os.path.join(INPUT_PATH_DIR, self.stream_function_filename )):  
            double_gyre.generate_streamfunction_values(**CONFIGS['TESTING']['GENERATE_STREAM_FUNCTION_FIELDS'] )
            
        if not( os.path.exists( os.path.join(DISCRETE_VELOCITY_PATH_DIR, self.velocity_filename )) 
        and os.path.exists( os.path.join(INTERP_VELOCITY_PATH_DIR, self.velocity_func_filename )) ):  
            generate_velocity_fields.generate_velocity_fields( **CONFIGS[ 'TESTING' ]['GENERATE_VELOCITY_FIELDS'] )

        print('Finished set up')
    
    def test_interpolated_velocity_field(self):
        """
        Compare interpolated double gyre velocity field values 
        with true double gyre velocity field values

        comparing functions: 
        double_gyre.update
        interp.velocity_update
        """ 
        uv_fullpath = os.path.join( DISCRETE_VELOCITY_PATH_DIR, self.velocity_filename) 
        u, v = util.load_velocity_field( uv_fullpath )
        uvinterp_dir_fullpath = os.path.join( INTERP_VELOCITY_PATH_DIR, self.velocity_func_filename)
        u_interp,v_interp = util.load_velocity_field( uvinterp_dir_fullpath )
        xct , yct = self.xct,self.yct
        reduced_xct, reduced_yct = util.calculate_xct_yct_ratio( xct , yct )
        # dx and dy should be equal
        dy = dx = float(min( reduced_xct, reduced_yct )) / (float( min( xct, yct ) ))
        vfuncs = (u_interp,v_interp)
        x = np.linspace(0.5*dx,float(reduced_xct)-(0.5*dx),xct )[::reduced_xct][1:-1]
        y = np.linspace(0.5*dy,float(reduced_yct)-(0.5*dy),yct )[::reduced_yct][1:-1]
        
        # test a state that was used in interpolation (should be NO difference)
        states = np.column_stack((x,y))
        for state in states:
            state_expanded_dim = np.expand_dims(state, axis=0)
            for t in np.arange(0,10,1):
                # print( state_expanded_dim )
                true_values = double_gyre.update(state_expanded_dim ,t, delta= dx   )
                interpolated_values = interp.velocity_update(vfuncs, state_expanded_dim, t, dt=self.dt, x_range=(0.0,reduced_xct), y_range=(0.0,reduced_yct)) 
                # print(true_values, interpolated_values)
                # self.assertEqual( true_values, interpolated_values )
                np.testing.assert_array_almost_equal(true_values,
                                                    interpolated_values,
                                                    decimal=10)

        # test interpolated states (should be SMALL difference)
        states = np.array( [ [random.uniform(dx/2,reduced_xct-(dx/2)), random.uniform(dy/2,reduced_yct-(dy/2))] for _ in range(10) ]  )
        for state in states:
            state_expanded_dim = np.expand_dims(state, axis=0)
            for t in np.arange(0,10,1):
                # print( state_expanded_dim )
                true_values = double_gyre.update(state_expanded_dim ,t, delta= dx   )
                interpolated_values = interp.velocity_update(vfuncs, state_expanded_dim, t, dt=self.dt, x_range=(dx/2,reduced_xct-(dx/2)), y_range=(dy/2,reduced_yct-(dy/2))) 
                # print(true_values, interpolated_values)
                # self.assertEqual( true_values, interpolated_values )
                np.testing.assert_array_almost_equal(true_values,
                                                    interpolated_values,
                                                    decimal=1)

        


if __name__ == '__main__':
    unittest.main()


