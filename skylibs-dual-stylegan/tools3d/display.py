from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np



def plotSubFigure(X, Y, Z, subfig, type_):
    fig = plt.gcf()
    ax = fig.add_subplot(1, 3, subfig, projection='3d')
    #ax = fig.gca(projection='3d')
    if type_ == "colormap":
        ax.plot_surface(X, Y, Z, cmap=cm.viridis, rstride=1, cstride=1,
                        shade=True, linewidth=0, antialiased=False)
    else:
        ax.plot_surface(X, Y, Z, color=[0.7, 0.7, 0.7], rstride=1, cstride=1,
                        shade=True, linewidth=0, antialiased=False)

    ax.set_aspect("equal")

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0 * 0.6
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    az, el = 90, 90
    if type_ == "top":
        az = 130
    elif type_ == "side":
        az, el = 40, 0

    ax.view_init(az, el)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plt.grid(False)
    plt.axis('off')


def plotDepth(Z):
    x = np.linspace(0, Z.shape[0]-1, Z.shape[0])
    y = np.linspace(0, Z.shape[1]-1, Z.shape[1])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 6))
    plotSubFigure(X, Y, Z, 1, "colormap")
    plotSubFigure(X, Y, Z, 2, "top")
    plotSubFigure(X, Y, Z, 3, "side")

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
