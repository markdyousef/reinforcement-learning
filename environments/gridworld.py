import numpy as np
import sys
from gym.envs.toy_text import discrete
from io import StringIO

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

class GridworldEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}
    def __init__(self, shape=[4,4]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be list/tuple of length 2')

        self.shape = shape

        nS = np.prod(shape)
        nA = 4
        P = {}

        MAX_Y, MAX_X = shape
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            P[s] = {a: [] for a in range(nA)}

            is_done = lambda s: s==0 or s==(nS-1)
            reward = 0.0 if is_done(s) else -1.0

            # stuck in terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            # not a terminal state
            else:
                ns_up = s if y==0 else s-MAX_X
                ns_right = s if x==(MAX_X-1) else s+1
                ns_down = s if y==(MAX_Y-1) else s+MAX_X
                ns_left = s if x==0 else s-1

                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]
            
            it.iternext()
        
        # initial state distribution is uniform
        isd = np.ones(nS) / nS

        # should not be used in model-free environments
        self.P = P

        super(GridworldEnv, self).__init__(nS, nA, P, isd)
    
    def _render(self, mode='human', close=False):
        if close: 
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        
        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s==s: # current
                output = ' x '
            elif s==0 or s==self.nS-1: # terminal state
                output = ' T '
            else:
                output = ' o '
            
            if x==0:
                output = output.lstrip()
            if x==self.shape[1]-1:
                output = output.rstrip()
            
            outfile.write(output)
            
            if x==self.shape[1]-1:
                outfile.write('\n')
            
            it.iternext()