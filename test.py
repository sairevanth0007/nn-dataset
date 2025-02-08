import unittest
from pprint import pprint

import ab.nn.api as api
from ab.nn.util.Const import default_nn_path, default_nn_name
from ab.nn.util.Util import read_py_file_as_string


class Testing(unittest.TestCase):
    def test_data(self):
        o = api.data(cast_prm=False)
        o2 = api.data(task='img-classification', nn=default_nn_name, cast_prm=False)
        print(o2)
        self.assertNotEqual(len(o2), 0)
        self.assertGreater(len(o), len(o2))

    def test_check_nn(self):
        code = read_py_file_as_string(default_nn_path)
        pprint(api.check_nn(code, 'img-classification', 'cifar-10', 'acc',
                            {'lr': 0.01, 'batch': 10, 'dropout': 0.2, 'momentum': 0.9,
                             'transform': 'norm_256_flip', 'epoch': 1},
                            save_to_db=False))


if __name__ == '__main__':
    unittest.main()
