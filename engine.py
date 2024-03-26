import numpy as np

MAX_TURN = 18*2*np.pi/360

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
    
    def __init__(self, game, strategy = default_strategy):
        
        self.id = Pod.nb
        Pod.nb += 1

        self.game = game
        self.strategy = strategy
        
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

        if action == None:
            action = self.strategy(self)
        turn, thrust = action

        #end of previous timestemp        
        self.v *= 0.85
        self.pos = np.array(self.pos).round()
        self.v = np.trunc(self.v)
        
        #update v and orientation
#        if "BOOST" == thrust:
#            if self.boost:
#                thrust = 650
#            else:
#                thrust = 100
#            self.boost = False
#        else:
        assert 0 <= thrust <= 100
        self.orientation += np.clip(turn, -MAX_TURN, MAX_TURN)
        self.orientation = mod(self.orientation)
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

        print("collide", self.dist(pod.pos))
        
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