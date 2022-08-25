import numpy as np

from . import EnvironmentMap


def test_imageCoordinates():
    for s in range(4, 9, 4):
        e = EnvironmentMap(np.zeros((s,s)), 'Angular')
        u, v = e.imageCoordinates()
        # All rows/columns are the same
        assert np.all(np.diff(u, axis=0) == 0)
        assert np.all(np.diff(v, axis=1) == 0)
        # All the columns/rows are spaced by 1/s
        assert np.all(np.diff(u, axis=1) - 1/s == 0)
        assert np.all(np.diff(v, axis=0) - 1/s == 0)
        # First element is (1/s)/2
        assert u[0,0] == v[0,0] == 1/s/2
