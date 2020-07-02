import unittest
import os 
import compare_trajectories 
from config import * 
import util 
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
            np.array([[0.2, 0.4], [0.5, 1.0], [0.7, 1.6]]),
            np.array([[0.2, 0.4], [0.5, 1.0], [0.7, 1.6]])
        ]
        total_scores, total_observations = np.array([]), np.array([])
        list_scores, list_observations = compare_trajectories.evaluate(list_of_state_vectors,max_lengths=[1.0, 2.0])
        total_scores = util.add_numpy_arrays(total_scores, list_scores)
        total_observations = util.add_numpy_arrays(total_observations, list_observations)
        print(total_scores)
        print(total_observations)



if __name__ == '__main__':
    unittest.main()