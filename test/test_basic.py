import unittest
import os
import pandas as pd
import textanalysis as ta

class TestExistence(unittest.TestCase):
    def setUp(self):
        self.text = pd.read_pickle(os.path.join(os.getcwd(), 'data', 'output', 'text_data.pkl'))

    def test_upper(self):
        self.assertEqual(self.text.dtypes[2].name, 'float64')

    def tearDown(self):
        self.text = None

if __name__ == '__main__':
    unittest.main()
