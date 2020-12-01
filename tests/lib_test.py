# -*- coding: UTF-8 -*-

# Import from standard library
import os
import toolboxpv
# Import from our lib
from toolboxpv.lib import show_results
import pytest


def test_show_results():
    y_test =[1, 2, 3]
    y_pred = [2, 3, 4]
    metric = 'mae'
    assert show_results(y_test, y_pred, metric) == 1.0
