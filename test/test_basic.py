import unittest
import os
import pandas as pd
from textanalysis import analysis

class TestExistence(unittest.TestCase):
    def setUp(self):
        self.text = pd.read_pickle(os.path.join(os.getcwd(), 'data', 'output', 'australia_governance.pkl'))

    def test_upper(self):
        self.assertEqual(self.text.dtypes[2].name, 'float64')

    def test_directory(self):
        self.assertEqual(os.path.exists(os.path.join(os.getcwd(), 'data', 'texts')), True)
        dir_path = os.fsencode(os.path.join(os.getcwd(), 'data', 'texts'))
        texts, filenames, languages = analysis.load_texts(dir_path)
        self.assertEqual(len(texts), 7) 
        self.assertEqual(len(filenames), 7)
        self.assertEqual(len(languages), 7)

    def test_pipeline(self):
        sclass, zclass = analysis.build_nlp_pipelines()
        self.assertEqual(sclass.task, 'sentiment-analysis')
        self.assertEqual(zclass.task, 'zero-shot-classification')
    
    def tearDown(self):
        self.text = None

if __name__ == '__main__':
    unittest.main()
