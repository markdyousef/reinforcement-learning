import matplotlib.pyplot as plt
from matplotlib.cm import coolwarm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# plots value function for blackjack env
def plot_value_function(V, title="Value Function"):
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x+1)
    y_range = np.arange(min_y, max_y+1)
    X, Y = np.meshgrid(x_range, y_range)

    # find value for all (x,y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X,Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1,
                                cstride=1, cmap=coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()
    
    plot_surface(X, Y, Z_noace, f'{title} (No Usable Ace)')
    plot_surface(X, Y, Z_ace, f'{title} (Usable Ace)')