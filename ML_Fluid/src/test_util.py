import unittest
import util 
import numpy as np 


class Test_util(unittest.TestCase):
    def test_flatten_time_series(self): 
        # check basic functionality
        test_array = np.random.rand(3, 4, 10)
        flattened_test_array = util.flatten_time_series( test_array )
        self.assertEqual( flattened_test_array.shape, (12,10)  )
        # check warning logging
        with self.assertLogs(level='INFO') as cm:
            test_array_reordered = np.random.rand(10, 3, 4)
            flattened_test_array_reordered = util.flatten_time_series( test_array_reordered )
            self.assertEqual( flattened_test_array.shape, (12,10)  )
        self.assertEqual(cm.output, [ 'WARNING:root:Last dimension is not the largest, check order of dimensions' ])

if __name__ == '__main__':
    unittest.main()