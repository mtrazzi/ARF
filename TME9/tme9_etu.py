import gym
import numpy as np
from collections import defaultdict
from ple.games.flappybird import FlappyBird
from ple import PLE
import gym_ple
import matplotlib.pyplot as plt



def test_gym():
    for game_name in ["Taxi-v2","FlappyBird-v0"]:
        game = gym.make(game_name)
        game.reset()
        r = 0
        for i in range(100):
            action = game.action_space.sample()
            observation,reward,done,info = game.step(action)
            r+=reward
            game.render()
            print("iter {} : action {}, reward {}, state {} ".format(i,action, reward,observation))
            if done:
                break
        print(" Succes : {} , reward cumulatif : {} ".format(done,r))
 





class Eve(object):
    def __init__(self,politique=None,nchoices=None):
        if nchoices is not None:
            self.reset(nchoices)
        self.politique = politique
    def reset(self,nchoices):
        self.nchoices = nchoices
        self.rewards = np.zeros(self.nchoices)
        self.times = np.zeros(self.nchoices)
    def getAction(self):
        if self.politique is None:
            return np.random.randint(self.nchoices)
        return self.politique(self.rewards,self.times)
    def setReward(self,action,r):
        self.rewards[action]+=r
        self.times[action]+=1

