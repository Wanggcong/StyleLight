import numpy as np
from scipy.sparse import coo_matrix, vstack as sparse_vstack
from scipy.sparse.linalg import lsqr as sparse_lsqr


from . import display
from . import spharm


def getMaskDerivatives(mask):
    """
    Build the derivatives of the input `mask`

    :returns: (`Mx`, `My`), containing the U-axis and V-axis derivatives of the mask.
    """
    idxmat = np.zeros_like(mask, np.int32)
    idxmat[mask] = np.arange(np.sum(mask))

    # Build the pixel list (for in-order iteration) and set (for O(1) `in` operator)
    pts = list(zip(*np.where(mask)))
    pts_set = set(pts)

    col_x, data_x = [], []
    col_y, data_y = [], []
    for x, y in pts:
        p0 = idxmat[x, y]
        # x-derivative
        if (x, y+1) in pts_set: # If pixel to the right
            pE = idxmat[x, y+1]
            col_x.extend([p0, pE]); data_x.extend([-1, 1])
        elif (x, y-1) in pts_set: # If pixel to the left
            pW = idxmat[x, y-1]
            col_x.extend([pW, p0]); data_x.extend([-1, 1])
        else: # Pixel has no right or left but is valid, so must have an entry
            col_x.extend([p0, p0]); data_x.extend([0, 0])

        # y-derivative
        if (x+1, y) in pts_set: # If pixel to the bottom
            pS = idxmat[x+1, y]
            col_y.extend([p0, pS]); data_y.extend([-1, 1])
        elif (x-1, y) in pts_set: # If pixel to the top
            pN = idxmat[x-1, y]
            col_y.extend([pN, p0]); data_y.extend([-1, 1])
        else: # Pixel has no right or left but is valid, so must have an entry
            col_y.extend([p0, p0]); data_y.extend([0, 0])

    nelem = np.sum(mask)

    row_x = np.tile(np.arange(len(col_x)//2)[:,np.newaxis], [1, 2]).ravel()
    row_y = np.tile(np.arange(len(col_y)//2)[:,np.newaxis], [1, 2]).ravel()
    Mx = coo_matrix((data_x, (row_x, col_x)), shape=(len(col_x)//2, nelem))
    My = coo_matrix((data_y, (row_y, col_y)), shape=(len(col_y)//2, nelem))

    return Mx, My


def NfromZ(surf, mask, Mx, My):
    """
    Compute (derivate) the normal map of a depth map.
    """

    normals = np.hstack((Mx.dot(surf.ravel())[:,np.newaxis], My.dot(surf.ravel())[:,np.newaxis], -np.ones((Mx.shape[0], 1), np.float64)))
    # Normalization
    normals = (normals.T / np.linalg.norm(normals, axis=1)).T

    # Apply mask
    out = np.zeros(mask.shape + (3,), np.float32)
    out[np.tile(mask[:,:,np.newaxis], [1, 1, 3])] = normals.ravel()

    return out


def ZfromN(normals, mask, Mx, My):
    """
    Compute (integrate) the depth map of a normal map.
    
    The reconstruction is up to a scaling factor.
    """
    b = -normals
    b[:,2] = 0
    b = b.T.ravel()

    N = normals.shape[0]
    ij = list(range(N))
    X = coo_matrix((normals[:,0], (ij, ij)), shape=Mx.shape)
    Y = coo_matrix((normals[:,1], (ij, ij)), shape=Mx.shape)
    Z = coo_matrix((normals[:,2], (ij, ij)), shape=Mx.shape)
    A = sparse_vstack((Z.dot(Mx),
                       Z.dot(My),
                       Y.dot(Mx) - X.dot(My)))
    # Is the 3rd constraint really useful?

    surf = sparse_lsqr(A, b)
    surf = surf[0]
    surf -= surf.min()

    out = np.zeros(mask.shape, np.float32)
    out[mask] = surf.ravel()

    return out


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from scipy.misc import imresize
    from scipy.ndimage.interpolation import zoom

    # Usage example
    # Step 1) a) Build a depth map ...
    surf = np.array([[1, 1, 1, 1],
                     [1, 2, 3, 1],
                     [2, 3, 5, 2],
                     [3, 3, 2, 3]], np.float32)


    # Step 1) b) ... and its mask
    mask = np.ones(surf.shape, np.bool)
    # Simulate mask, uncomment to test
    mask[1,1] = 0
    # mask[1,2] = 0
    # mask[0,2] = 0
    # mask[3,1] = 0

    # Step 1) c) Scale it up to spice up life
    surf = zoom(surf, (5, 5), order=1)
    mask = zoom(mask, (5, 5), order=0)
    surf[~mask] = 0
    surf_ori = surf.copy()
    surf = surf[mask]

    # Step 2) Compute the mask derivatives
    Ms = getMaskDerivatives(mask)

    # Step 3) Compute the normal map from the depth map
    normals = NfromZ(surf, mask, *Ms)

    # Step 4) Compute the depth map from the normal map
    masked_normals = normals[np.tile(mask[:,:,np.newaxis], [1, 1, 3])].reshape([-1, 3])
    surf_recons = ZfromN(masked_normals, mask, *Ms)

    # Visualize the results
    plt.subplot(131); plt.imshow(surf_ori, interpolation='nearest'); plt.colorbar()
    plt.subplot(132); plt.imshow((normals+1)/2, interpolation='nearest')
    plt.subplot(133); plt.imshow(surf_recons, interpolation='nearest'); plt.colorbar()
    plt.show()
