"""
@author : Tien Nguyen
@date   : 2023-03-06
"""
import unittest

import numpy
import torch

import utils

class TestStringMethods(unittest.TestCase):
    def test_cal_acc_case_1(
            self
        ) -> None:
        labels = torch.tensor([
            [10, 11, 1, 0, 0],
            [9,  5,  0, 0, 0],
            [7,  3,  1, 1, 1],
            [6,  2,  0, 0, 0],
            [2,  1,  1, 1, 1],
        ])
        preds = torch.tensor([
            [10, 9,  1, 0, 1],
            [7,  5,  1, 1, 1],
            [6,  2,  0, 0, 1],
            [4,  1,  0, 0, 0],
            [2,  0,  0, 1, 0],
        ])
        result = utils.cal_acc(preds, labels).cpu().detach().item()
        expected = 0.4
        self.assertEqual(numpy.round(result, 2), expected)

    def test_cal_acc_case_2(
            self
        ) -> None:
        labels = torch.tensor([
            [10, 11, 1, 0, 0],
            [9,  5,  0, 0, 0],
            [7,  3,  1, 1, 1],
            [6,  2,  0, 0, 0],
            [2,  1,  1, 1, 1],
            [3,  5,  0, 0, 1]
        ])
        preds = torch.tensor([
            [10, 9,  1, 0, 1],
            [7,  5,  0, 1, 1],
            [6,  2,  0, 0, 1],
            [4,  1,  0, 0, 0],
            [2,  0,  0, 1, 0],
            [3,  6,  1, 1, 1]
        ])
        result = utils.cal_acc(preds, labels).cpu().detach().item()
        expected = 0.52
        self.assertEqual(numpy.round(result, 2), expected)

if __name__ == '__main__':
    unittest.main()
