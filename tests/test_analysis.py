import unittest
from app import analyze
from unittest.mock import patch
from app import ocr_executable

class AnalysisTests(unittest.TestCase):
    def payload(self, text):
        return dict(csv=text, confirmed=True, low=1, high=5,
                    scales=[dict(name='A', items=['a','b'], reverse=['b']),
                            dict(name='B', items=['c'], reverse=[])])

    def test_reverse_alpha_missing_and_pairwise(self):
        result = analyze(self.payload('a,b,c\n1,5,1\n2,4,2\n3,3,3\n4,,4\n'))
        a = result['summaries'][0]
        self.assertEqual(a['n'], 3)
        self.assertEqual(a['excluded'], 1)
        self.assertEqual(a['missing_cells'], 1)
        self.assertAlmostEqual(a['mean'], 2)
        self.assertAlmostEqual(a['sd'], 1)
        self.assertAlmostEqual(a['alpha'], 1)
        self.assertEqual(result['correlations'][0]['n'], 3)
        self.assertAlmostEqual(result['correlations'][0]['r'], 1)

    def test_reject_invalid_observation(self):
        for value in ['6','text','inf','1.5']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                analyze(self.payload(f'a,b,c\n{value},1,1\n2,2,2\n3,3,3'))

    def test_reject_duplicate_headers(self):
        with self.assertRaises(ValueError):
            analyze(self.payload('a,a,c\n1,1,1\n2,2,2\n3,3,3'))

    def test_constant_variable(self):
        result = analyze(self.payload('a,b,c\n1,5,1\n1,5,1\n1,5,1'))
        self.assertIsNone(result['summaries'][0]['alpha'])
        self.assertIsNone(result['correlations'][0]['r'])

    def test_confirmation_required(self):
        data=self.payload('a,b,c\n1,1,1\n2,2,2\n3,3,3')
        data['confirmed']=False
        with self.assertRaises(ValueError):
            analyze(data)

    def test_missing_ocr_engine_is_actionable(self):
        with patch('app.Path.is_file', return_value=False), patch('app.shutil.which', return_value=None):
            with self.assertRaisesRegex(ValueError, 'OCR 엔진'):
                ocr_executable()

if __name__ == '__main__':
    unittest.main()
