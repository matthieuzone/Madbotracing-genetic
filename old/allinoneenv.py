import numpy as np
import pygame
import gym
from gym import spaces
import pygui

MAX_TURN = 18*2*np.pi/360
RENDER_FPS = 25
N_DT = 1
TURNS_TO_WIN = 3

DEF_MAP = np.array([[10554, 5986], [3566, 5151], [13559, 7611], [12479, 1364]])

OBS_SPACE = spaces.Dict({
    #"p": spaces.Box(-2000, 18000, shape=(2,)),
    #"v": spaces.Box(-2000, 2000, shape=(2,)),
    #"next_checkpoint" : spaces.Box(0, 16000, shape=(2,)),
    "v" : spaces.Box(0, 1300, shape=(1,)),
    "v_dir" : spaces.Box(-np.pi, np.pi, shape=(1,)),
    "next_checkpoint_dist" : spaces.Box(0, 20000, shape=(1,)),
    "next_checkpoint_angle" : spaces.Box(-np.pi, np.pi, shape=(1,)),
    #"nbcp" : spaces.Box(0, 20, shape=(1,)),
})
OBS_SPACE = spaces.flatten_space(OBS_SPACE)

ACT_SPACE = spaces.Dict({
    "turn": spaces.Box(-np.pi, np.pi, shape=(1,)),
    "thrust": spaces.Box(0, 100, shape=(1,))
})

ACT_SPACE = spaces.flatten_space(ACT_SPACE)

def u(theta):
    return np.array((np.cos(theta), np.sin(theta)))

def dir(v):
    if np.linalg.norm(v) == 0:
        return 0
    v = v / np.linalg.norm(v)
    if v[1] == 0:
        return -(v[0]/abs(v[0]) - 1)*np.pi/2
    return np.arccos(v[0]) * v[1]/abs(v[1])

def mod(theta):
    return ((theta + np.pi)  % (2*np.pi)) - np.pi


def default_strategy(self):
    return self.game.checkpoints[self.nextcp], 100


class Pod:
     
    nb = 0
    
    def __init__(self, game):
        
        self.id = Pod.nb
        Pod.nb += 1

        self.game = game
        
        self.pos = np.array([0.,0.])
        self.v = np.array([0.,0.])
        self.orientation = 0
        self.nextcp = 0
        self.nbcp = 0
        self.nb_turns = 0
        self.boost = True

    def dist(self,p):
        return np.linalg.norm(p - self.pos)

    def timestep(self, action = None):

        turn, thrust = action

        #end of previous timestemp        
        self.v *= 0.85
        self.pos = np.array(self.pos).round()
        self.v = np.trunc(self.v)
        
        #update v and orientation
        turn = np.clip(turn, -MAX_TURN, MAX_TURN)

#        if "BOOST" == thrust:
#            if self.boost:
#                thrust = 650
#            else:
#                thrust = 100
#            self.boost = False
#        else:
        assert 0 <= thrust <= 100
        self.orientation = mod(self.orientation + turn)
        self.v += thrust*u(self.orientation)
                
    def move(self, Ndt):
        self.pos += self.v/Ndt
        #self.checkCollide()
        if self.dist(self.game.checkpoints[self.nextcp]) < 600:
            self.on_checkpoint()

    def on_checkpoint(self):
        self.nextcp += 1
        self.nbcp += 1
        self.nextcp = self.nextcp % len(self.game.checkpoints)
        if self.nextcp == 0:
            self.nb_turns += 1
        
    def collide(self, pod):       
        u = pod.pos - self.pos
        u = u/np.linalg.norm(u)
        v1p = np.dot(self.v, u)*u
        v2p = np.dot(pod.v, u)*u
        p = v2p - v1p
        if np.linalg.norm(p) < 120:
            p *= 120/np.linalg.norm(p)      
        self.v += p
        pod.v -= p
    
    def checkCollide(self):
        for pod in self.game.pods:
            if pod.id != self.id:
                if self.dist(pod.pos) < 800:
                    self.collide(pod)

class Env():

    def __init__(self, render_mode = None, reset_checkpoints = True, nb_checkpoints = 4, max_steps = 100):

        self.observation_space = OBS_SPACE
        self.action_space = ACT_SPACE

        #Define parameters
        self.reset_checkpoints = reset_checkpoints
        self.nb_checkpoints = nb_checkpoints
        self.render_mode = render_mode
        self.max_steps = max_steps

        #initialize
        self.checkpoints = DEF_MAP
        self.reset()

    def _get_obs(self):
        return (
            np.linalg.norm(self.player.v) / 1300,
            dir(self.player.v) / np.pi,
            self.player.dist(self.checkpoints[self.player.nextcp]) / 10000,
            (dir(self.checkpoints[self.player.nextcp] - self.player.pos) - self.player.orientation) / np.pi,
        )
    
    def reset(self, render_mode = "idem"):
        if render_mode != "idem":
            self.render_mode = render_mode
        if self.reset_checkpoints:             
            self.checkpoints = [np.random.randint(1500, 7500, (2,)) for i in range (self.nb_checkpoints)]

        self.player = Pod(self)
        self.pods = [self.player]
        self.window = None
        self.clock = None        
        self.nb_steps = 0

        return self._get_obs()

    def step(self, action):   
        self.nb_steps += 1
        #for pod in self.pods:
        self.player.timestep(action)
        for i in range(N_DT):
                self.player.move(N_DT)
                if self.render_mode == "human":
                    pygui.render_frame(self)
        return self._get_obs(), self.terminated()
    
    def reward(self):
        d = self.player.dist(self.checkpoints[self.player.nextcp])
        prc = 1 - d / np.linalg.norm(self.checkpoints[self.player.nextcp] - self.checkpoints[self.player.nextcp - 1])
        return self.player.nbcp + max(0, prc)
    
    def terminated(self):
        return self.nb_steps >= self.max_steps
