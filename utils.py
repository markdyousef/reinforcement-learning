import matplotlib.pyplot as plt
from matplotlib.cm import coolwarm
import numpy as np
from collections import namedtuple
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])

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

def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # plot episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)
    
    # plot episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smooth = pd.Series(stats.episode_rewards).rolling(smoothing_window,
                                                              min_periods=smoothing_window).mean()
    plt.plot(rewards_smooth)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)
    
    # plot time steps and episode number
    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episodes")
    plt.title("Episode per Time Step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)
    
    return fig1, fig2, fig3
    