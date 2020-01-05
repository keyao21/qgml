import unittest
import os 
import compare_trajectories 
from config import * 


class Test_compare_trajectories(unittest.TestCase):
    def test_generate_single_trajectory(self): 
        test_velocity_func_filenames = ["dguv.128.64.est.interp", "dguv.128.64.actual.interp"]
        test_velocity_func_fullpaths = [os.path.join(INTERP_VELOCITY_PATH_DIR, test_velocity_func_filename) \
                                        for test_velocity_func_filename in test_velocity_func_filenames]
        initial_conditions = [0.2,0.4]
        elapsed_time = 100
        dt = 0.1
        trajectory = compare_trajectories.generate_single_trajectory(velocity_func_fullpaths=test_velocity_func_fullpaths, 
                                                                    initial_conditions=initial_conditions, 
                                                                    elapsed_time=elapsed_time, 
                                                                    dt=dt)
        print( trajectory ) 
        print( trajectory.shape )

if __name__ == '__main__':
    unittest.main()