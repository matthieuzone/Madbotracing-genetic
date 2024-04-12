import tensorflow as tf
from keras import layers
import numpy as np

from environement.maps import MAPS
import actor

class Batch:

    def __init__(self, env, mutation_rate, reproduction_rate, keep_each_gen, actors, runs_for_eval = 1, reset_checkpoints = True, modelinit = actor.create_model):
        #define parameters
        self.runs_for_eval = runs_for_eval
        self.mutation_rate = mutation_rate
        self.reproduction_rate = reproduction_rate
        self.keep_each_gen = keep_each_gen
        self.actors = actors
        self.env = env
        self.reset_checkpoints = reset_checkpoints

        self.population = [actors(modelinit(env.observation_space.shape[0], env.action_space.shape[0])) for i in range(keep_each_gen)]

        self.gen = 0

    def reproduce(self):
        next_gen = []
        for model in self.population:
            next_gen.append(model)
            for i in range(self.reproduction_rate):
                child = model.copy()
                child.mutate(self.mutation_rate)
                next_gen.append(child)
        return next_gen

    def next_gen(self):
        newpop = self.reproduce()
        maps = [MAPS[np.random.randint(0,len(MAPS))] for i in range(self.runs_for_eval)]
        score = [0] * len(newpop)
        if self.reset_checkpoints:
            for map in maps:
                self.env.set_map(map)
                for i in range(len(newpop)):
                    score[i] += (newpop[i].play(self.env))/len(maps)
        else:
            for i in range(len(newpop)):
                score[i] += (newpop[i].play(self.env))
        best = np.argsort(score)[-self.keep_each_gen:]
        self.population = [newpop[b] for b in best]
        self.gen += 1
        print("gen : ", self.gen, " best score : ", score[best[-1]], "avg score :", np.mean(score))

    def train(self, N):
        for i in range(N):
            self.next_gen()

    def get_best(self):
        score = []
        for model in self.population:
            score.append(model.eval(self.env, self.runs_for_eval))
        return self.population[np.argmax(score)]
    
    def savePop(self, path):
        for i in range(len(self.population)):
            self.population[i].model.save(path + "/" + str(i))
        
    def loadPop(self, path, n = None):
        if n == None:
            n = self.keep_each_gen
        for i in range(n):
            a = self.actors(tf.keras.models.load_model(path + "/" + str(i)))
            self.population.append(a)
        
