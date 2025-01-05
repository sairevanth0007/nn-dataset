import unittest

import ab.nn.api as api


class Testing(unittest.TestCase):
    def test_api(self):
        o = api.data()
        print(o)
        self.assertIsNotNone(o)


if __name__ == '__main__':
    unittest.main()
