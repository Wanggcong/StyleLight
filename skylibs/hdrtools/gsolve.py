# Taken from Debevec1997 "Recovering High Dynamic Range Radiance Maps
#                         from Photographs"
#
# gsolve.py - Solve for imaging system response function
#
# Given a set of pixel values observed for several pixels in several
# images with different exposure times, this function returns the
# imaging system's response function g as well as the log film irradiance
# values for the observed pixels.
#
# Assumes:
#
# Zmin = 0
# Zmax = 255
#
# Arguments:
#
# Z(i,j) is the pixel values of pixel location number i in image j
# B(j) is the log delta t, or log shutter speed, for image j
# l is lamdba, the constant that determines the amount of smoothness
# w(z) is the weighting function value for pixel value z
#
# Returns:
#
# g(z) is the log exposure corresponding to pixel value z
# lE(i) is the log film irradiance at pixel location i
#

import numpy as np


def gsolve(Z, B, l, w):
    n = 256
    A = np.zeros((Z.shape[0]*Z.shape[1] + n - 1, n + Z.shape[0]), dtype=float)
    b = np.zeros((A.shape[0], 1), dtype=float)

    # Include the data-fitting equations
    k = 0;
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            wij = w[Z[i,j]]
            A[k, Z[i,j]] = wij
            A[k, n + i] = -wij 
            b[k, 0] = wij * B[j]
            k += 1

    # Fix the curve by setting its middle value to 0
    A[k, 128] = 1
    k += 1

    # Include the smoothness equations
    for i in range(n - 2):
        A[k, i+0] = l*w[i+1]
        A[k, i+1] = -2*l*w[i+1]
        A[k, i+2] = l*w[i+1]
        k += 1

    # Solve the system using SVD
    x = np.linalg.lstsq(A, b)[0]
    g = x[:n]
    lE = x[n:]

    return g, lE


def weights(z_min=0, z_max=255):
    """Outputs the weights of the z(i) input pixels.
    This is a direct implementation of eq. 4."""
    z = np.array(range(z_min, z_max + 1), dtype='float')

    lower = z <= 0.5*(z_min + z_max)
    upper = z > 0.5*(z_min + z_max)
    z[lower] = z[lower] - z_min
    z[upper] = z_max - z[upper]
    z /= z.max()

    return z
