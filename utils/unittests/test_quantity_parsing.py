import json
import unittest
import sys 
sys.path.append('..')
from off_mgt import convert_quantity


class TestConvert(unittest.TestCase):


    def setUp(self):
        with open('test_quantity_parsing.json', 'r') as f:
            self.test_data = json.load(f)


    def test_convert(self):

        for s in self.test_data.keys():
            s_converted = convert_quantity(s)
            s_compare = {'val': s_converted['val'],
                         'unit': s_converted['unit']}
            try:
                self.assertEqual(s_compare, self.test_data[s])
            except AssertionError:
                print('Not correctly handled :', s)
                print('converted dict is:', s_compare)
                print('Correct dict is:', self.test_data[s])
