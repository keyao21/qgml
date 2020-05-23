import unittest
import util 
import numpy as np 



class Test_util(unittest.TestCase):
    def test_calculate_xct_yct_ratio(self): 
        self.assertEqual( (util.calculate_xct_yct_ratio(128, 64)), (2,1) )
        self.assertEqual( (util.calculate_xct_yct_ratio(64, 64)), (1,1) )
        self.assertEqual( (util.calculate_xct_yct_ratio(64, 128)), (1,2) )
    def test_stack_numpy_arrays(self): 
        a = np.array([1,2, 0 ])
        b = np.array([11,12,13])
        res = util.stack_numpy_arrays(a,b)
        print( res ) 
        np.testing.assert_array_equal( res, np.array([[1,2,0], [11,12,13]]))
        

        test_arr = np.array([ [1,  2, 0], 
                   [11,12,13],
                   [22,23,24]
                ])
        np.testing.assert_array_equal(util.stack_numpy_arrays(res,np.array([22,23,24])), test_arr)



if __name__ == '__main__':
    unittest.main()