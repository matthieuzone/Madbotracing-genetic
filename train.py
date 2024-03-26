from genetic import Batch
from wrappers import *
from environement import Environement

env = neurable(progressReward(TimeLimit(Environement(reset_checkpoints = False), 100)))

b = Batch(env, 0.2, 10, 20)

b.train(40)

b.savePop("saved/b.json")

b.env = neurable(progressReward(TimeLimit(Environement(reset_checkpoints = True), 150)))
b.runs_for_eval = 25


for i in range(10):
    b.train(4)
    b.savePop("saved/b.json")

b.env = neurable(Environement(reset_checkpoints = True))

for i in range(10):
    b.mutation_rate = 0.2*0.5**i
    b.train(5)
    b.savePop("saved/b.json") 