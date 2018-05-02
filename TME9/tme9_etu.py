import gym
import numpy as np
from collections import defaultdict
from ple.games.flappybird import FlappyBird
from ple import PLE
import gym_ple
import matplotlib.pyplot as plt
import time
from subprocess import call

def test_taxi(eve):
    for game_name in ["Taxi-v2"]:
        game = gym.make(game_name)
        game.reset()
        r = 0
        last_reward, last_action = 0, 0
        for i in range(1000):
            action = no_block(game.action_space, last_reward, last_action)
            print('action is ', action)
            observation,reward,done,info = game.step(action)
            try :
                game.render()
                r+=reward
            except:
                r-=reward
            last_reward, last_action = reward, action
            input("Press Enter to continue...")
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

def uniforme(action_space):
    print("action space: |", action_space, "|")
    return action_space.sample()

def only_move(action_space):
    return 3

def no_block(action_space, last_reward, last_action):
    if last_reward < 0:
        return action_space.sample()
    else:
        return last_action


def main():
    eve = Eve()
    test_taxi(eve)


if __name__ == '__main__':
    main()
