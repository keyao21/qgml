import unittest
import os 
import compare_trajectories 
from config import * 
import numpy as np 

class Test_compare_trajectories(unittest.TestCase):
    # def test_generate_single_trajectory(self): 
    #     test_velocity_func_filenames = ["dguv.128.64.est.interp", "dguv.128.64.actual.interp"]
    #     test_velocity_func_fullpaths = [os.path.join(INTERP_VELOCITY_PATH_DIR, test_velocity_func_filename) \
    #                                     for test_velocity_func_filename in test_velocity_func_filenames]
    #     initial_conditions = [0.2,0.4]
    #     elapsed_time = 100
    #     dt = 0.1
    #     trajectory = compare_trajectories.generate_single_trajectory(velocity_func_fullpaths=test_velocity_func_fullpaths, 
    #                                                                 initial_conditions=initial_conditions, 
    #                                                                 elapsed_time=elapsed_time, 
    #                                                                 dt=dt)
    #     print( trajectory ) 
    #     print( trajectory.shape )


    def test_evaluate(self): 
        list_of_state_vectors = [
            np.array([0.24, 0.46 , 0.86, 1.23, 1.44, 1.89, 1.23, 1.03, 0.87, 0.67, 0.76, 0.91, 1.34]),
            np.array([0.24, 0.46 , 0.86, 1.23, 1.44, 1.89, 1.23, 1.03, 0.87, 0.67, 0.76, 0.91, 1.34])
        ]
        res = compare_trajectories.evaluate(list_of_state_vectors,length=2.0)
        print(res)



if __name__ == '__main__':
    unittest.main()