"""
    "Example 5.1: Blackjack" from Reinforcement Learning: An Introduction (2nd Edition)
    Based on https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/blackjack.py
"""
import gym
from gym import spaces
from gym.utils import seeding

# (1)A, (2-10)Number cards, (10)J/Q/K 
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

def compare(a,b):
    return int((a>b))-int((a<b))

def draw_card(np_random):
    return np_random.choice(deck)

def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]

def usable_ace(hand):
    return 1 in hand and sum(hand)+10<=21

def sum_hand(hand):
    if usable_ace(hand):
        return sum(hand)+10
    return sum(hand)

def is_bust(hand):
    return sum_hand(hand)>21

def score(hand):
    return 0 if is_bust(hand) else sum_hand(hand)

class BlackjackEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        if action: # hit
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else: #stick
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = compare(score(self.player), score(self.dealer))
        return self._get_obs(), reward, done, {}
    
    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def _reset(self):
        self.player = draw_hand(self.np_random)
        self.dealer = draw_hand(self.np_random)

        # auto-draw additional card if score<12
        while sum_hand(self.player) < 12:
            self.player.append(draw_card(self.np_random))
        return self._get_obs()