import numpy as np
from numpy import tan, arctan, arccos, sqrt


def tetrahedronSolidAngle(a, b, c, lhuillier=True):
    """ Computes the solid angle subtended by a tetrahedron.
       
          omega = tetrahedronSolidAngle(a, b, c)
       
        The tetrahedron is defined by three vectors (a, b, c) which define the
        vertices of the triangle with respect to an origin.
       
        For more details, see:
          http://en.wikipedia.org/wiki/Solid_angle#Tetrahedron
       
        Both methods are implemented, but L'Huillier (default) is easier to
        parallelize and thus much faster. 
       
        ----------
        Jean-Francois Lalonde
    """
    assert a.shape[0] == 3, 'a must be a 3xN matrix'
    assert b.shape[0] == 3, 'b must be a 3xN matrix'
    assert c.shape[0] == 3, 'c must be a 3xN matrix'

    if lhuillier:
        theta_a = arccos(np.sum(b*c, 0))
        theta_b = arccos(np.sum(a*c, 0))
        theta_c = arccos(np.sum(a*b, 0))

        theta_s = (theta_a + theta_b + theta_c) / 2

        product = tan(theta_s/2) * tan((theta_s-theta_a) / 2) \
                  * tan((theta_s-theta_b) / 2) * tan((theta_s-theta_c) / 2)
        
        product[product < 0] = 0;
        omega = 4 * arctan( sqrt(product) )
    else:
        raise NotImplementedError()

    return omega
