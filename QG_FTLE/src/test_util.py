import unittest
import util 

class Test_util(unittest.TestCase):
    def test_calculate_xct_yct_ratio(self): 
        self.assertEqual( (util.calculate_xct_yct_ratio(128, 64)), (2,1) )
        self.assertEqual( (util.calculate_xct_yct_ratio(64, 64)), (1,1) )
        self.assertEqual( (util.calculate_xct_yct_ratio(64, 128)), (1,2) )

if __name__ == '__main__':
    unittest.main()