
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from engine import dir,mod

class FlatO(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.old_space = env.observation_space
        self.observation_space = spaces.flatten_space(env.observation_space)

    def observation(self, obs):
        return spaces.flatten(self.old_space, obs)
    
class FlatA(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.old_space = env.action_space
        self.action_space = spaces.flatten_space(env.action_space)

    def action(self, act):
        return spaces.unflatten(self.old_space, act)
    
class NormA(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.action_space.is_bounded()
        self.low = env.action_space.low
        self.high = env.action_space.high

        self.mean = (self.low + self.high)/2
        self.dev = self.high - self.low

        self.action_space = spaces.Box(low = -1, high = 1, shape=env.action_space.shape)
   
    def action(self, act):
        return self.dev*act/2 + self.mean
    
class NormO(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.observation_space.is_bounded()
        self.low = env.observation_space.low
        self.high = env.observation_space.high

        self.mean = (self.low + self.high)/2
        self.dev = self.high - self.low

        self.observation_space = spaces.Box(low = -1, high = 1, shape=env.observation_space.shape)
   
    def observation(self, obs):
        return 2*(obs - self.mean)/self.dev
    
def neurable(env):
    return NormA(NormO(FlatA(FlatO(env))))

class TimeLimit(gym.Wrapper):

    def __init__(self, env, limit):
        super().__init__(env)
        self.limit = limit

    def terminated(self):
        return self.limit <= self.env.unwrapped.nb_steps or self.env.unwrapped.terminated()
    
    def step(self, action):
        state, reward, done, truncated, info = self.env.unwrapped.step(action)
        return state, reward, done or self.limit <= self.env.unwrapped.nb_steps, truncated, info

class progressReward(gym.RewardWrapper):

    def reward(self, reward):
        d = self.env.unwrapped.player.dist(self.env.unwrapped.checkpoints[self.env.unwrapped.player.nextcp])
        prc = 1 - d / np.linalg.norm(self.env.unwrapped.checkpoints[self.env.unwrapped.player.nextcp] - self.env.unwrapped.checkpoints[self.env.unwrapped.player.nextcp - 1])
        return self.env.unwrapped.player.nbcp + max(0, prc)

class cibleAction(gym.ActionWrapper):
  
    def __init__(self, env):
        super().__init__(env)
        self.action_space = ACT_SPACE = spaces.Dict({
            "cible": spaces.Box(-2000, 18000, shape=(2,)),
            "thrust": spaces.Box(0, 100, shape=(1,))
        })

    def action(self, act):
        return {"trun" : mod(dir(act["cible"] - self.env.unwrapped.player.pos) - self.env.unwrapped.player.orientation), "thrust" : act["thrust"]}

class progressScore(gym.Wrapper):

    def score(self):
        player = self.env.unwrapped.player
        d = player.dist(self.env.unwrapped.checkpoints[player.nextcp])
        prc = 1 - d / np.linalg.norm(self.env.unwrapped.checkpoints[player.nextcp] - self.env.unwrapped.checkpoints[player.nextcp - 1])
        if player.dist(self.env.unwrapped.checkpoints[player.nextcp - 1]) <= 600 or prc < 0:
            prc =  (np.pi - abs(mod(player.orientation - dir(self.env.unwrapped.checkpoints[player.nextcp] - player.pos))))/100
        return player.nbcp + max(0, prc)

class relativeObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "v" : spaces.Box(0, 1300, shape=(1,)),
            "v_dir" : spaces.Box(-np.pi, np.pi, shape=(1,)),
            "next_checkpoint_dist" : spaces.Box(0, 20000, shape=(1,)),
            "next_checkpoint_angle" : spaces.Box(-np.pi, np.pi, shape=(1,)),
        })
   
    def observation(self, obs):
        return {
            'v' : np.linalg.norm(obs['v']),
            'v_dir' : dir(obs['v']) - self.env.unwrapped.player.orientation,
            "next_checkpoint_dist" : obs["next_checkpoint_dist"],
            "next_checkpoint_angle" : obs["next_checkpoint_angle"],
        }
    
class relativeObs2(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "v" : spaces.Box(0, 1300, shape=(1,)),
            "v_dir" : spaces.Box(-np.pi, np.pi, shape=(1,)),
            "next_checkpoint_dist" : spaces.Box(0, 20000, shape=(1,)),
            "next_checkpoint_angle" : spaces.Box(-np.pi, np.pi, shape=(1,)),
            "cp+1_dist" : spaces.Box(0, 20000, shape=(1,)),
            "cp+1_angle" : spaces.Box(-np.pi, np.pi, shape=(1,)),
        })
   
    def observation(self, obs):
        return {
            'v' : np.linalg.norm(obs['v']),
            'v_dir' : dir(obs['v']) - self.env.unwrapped.player.orientation,
            "next_checkpoint_dist" : obs["next_checkpoint_dist"],
            "next_checkpoint_angle" : obs["next_checkpoint_angle"],
            "cp+1_dist" : obs["cp+1_dist"],
            "cp+1_angle" : obs["cp+1_angle"],
        }