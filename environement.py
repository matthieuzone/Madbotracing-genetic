from engine import *
import pygame
import gymnasium as gym
from gymnasium import spaces
import pygui
from maps import MAPS


N_DT = 1
MAX_TURNS = 3

DEF_MAP = np.array([[10554, 5986], [3566, 5151], [13559, 7611], [12479, 1364]])

OBS_SPACE = spaces.Dict({
    "p": spaces.Box(-2000, 18000, shape=(2,)),
    "v": spaces.Box(-2000, 2000, shape=(2,)),
    "next_checkpoint" : spaces.Box(0, 16000, shape=(2,)),
    "next_checkpoint_dist" : spaces.Box(0, 20000, shape=(1,)),
    "next_checkpoint_angle" : spaces.Box(-np.pi, np.pi, shape=(1,)),
    #"nbcp" : spaces.Box(0, 20, shape=(1,)),
})

ACT_SPACE = spaces.Dict({
    "turn" : spaces.Box(-np.pi, np.pi, shape=(1,)),
    "thrust": spaces.Box(0, 100, shape=(1,))
})

class Environement(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, render_mode = None):
        
        self.observation_space = OBS_SPACE
        self.action_space = ACT_SPACE

        #Define parameters
        self.set_render_mode(render_mode)

        #initialize
        self.checkpoints = DEF_MAP
        self.reset()

    def set_map(self, map):
        self.checkpoints = map

    def set_render_mode(self, render_mode):        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return {
            #"next_checkpoint" : self.checkpoints[self.player.nextcp],
            "next_checkpoint_dist" : self.player.dist(self.checkpoints[self.player.nextcp]),
            #"nbcp" : self.player.nbcp,
            "p": self.player.pos,
            "v" : self.player.v,
            "next_checkpoint_angle" : mod(dir(self.checkpoints[self.player.nextcp] - self.player.pos) - self.player.orientation),
            "cp+1_dist" : self.player.dist(self.checkpoints[(self.player.nextcp + 1) % len(self.checkpoints)]),
            "cp+1_angle" : mod(dir(self.checkpoints[self.player.nextcp] - self.player.pos) - self.player.orientation),
        }
        
    def _get_info(self):
        return
    
    def reset(self, seed=None, options=None):

        super().reset(seed=seed, options=options)       
        self.player = Pod(self)
        self.pods = [self.player]
        self.window = None
        self.clock = None        
        self.nb_steps = 0

        return self._get_obs(), self._get_info()    

    def step(self, action):   
        self.nb_steps += 1
        for pod in self.pods:
            pod.timestep((action["turn"][0], action["thrust"][0]))
            for i in range(N_DT):
                    pod.move(N_DT)
                    if self.render_mode == "human":
                        pygui.render_frame(self)
            return self._get_obs(), self.reward(), self.terminated(), False, self._get_info()
    
    def reward(self):
        return
    
    def score(self):
        return -self.nb_steps
    
    def terminated(self):
        return self.player.nb_turns >= MAX_TURNS

    def render(self):
        if self.render_mode == "rgb_array":
            return pygui.render_frame(self)

    
