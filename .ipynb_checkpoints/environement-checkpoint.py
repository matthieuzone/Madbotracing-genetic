from engine import *
import gym

class Environement(gym.E):

    def __init__(self):

        self.observation_space = gym.spa
        
        self.checkpoints = [np.random.randint(1500, 7500, (2,)) for i in range (1)]        
        self.pod = Pod(self)
        self.pods = []

    def reset(self):        
        self.checkpoints = [np.random.randint(1500, 7500, (2,)) for i in range (1)]
        self.pod = Pod(self)
        return self.observe()

    def observe(self):
        a = [self.pod.pos, self.pod.v] + self.checkpoints[self.pod.nextchx :] + self.checkpoints[self.pod.nextchx :] + [(self.pod.dist(self.checkpoints[self.pod.nextchx]), dir(self.checkpoints[self.pod.nextchx] - self.pod.pos))]
        a = np.array(a)

        return a.flatten()
    
    def step(self, action):
            action = action[0]
            self.pod.timestep(((action[0], action[1]), action[2]))
            for i in range(Ndt):
                    self.pod.move()
            return self.observe(), -self.pod.dist(self.checkpoints[0]), self.pod.nextchx >= 1