import engine
import numpy as np
from keras.saving import load_model
import matplotlib.pyplot as plt

from environement import Environement
from wrappers import *

def strat(obs):
    goal = obs["next_checkpoint"] - obs["p"]
    next_checkpoint_dist = np.linalg.norm(obs["next_checkpoint"] - obs["p"])
    next_checkpoint_angle = engine.dir(obs["next_checkpoint"] - obs["p"])

    v = obs["v"]
    corr = v - (np.dot(v, goal) / np.dot(goal,goal))*goal
    aim = np.rint(v - 2*corr + obs["p"]).astype(int)    
    
    if abs(next_checkpoint_angle) > 85:
        return obs["next_checkpoint"], 0

    elif abs(next_checkpoint_angle) > 40 and next_checkpoint_dist < 1000:
        return obs["next_checkpoint"], 0

    
    elif next_checkpoint_dist <= 1200 and np.dot(v,goal) > np.linalg.norm(goal)*next_checkpoint_dist/4:
        return obs["next_checkpoint"], 0

    elif np.dot(v,goal) < 0.5*np.linalg.norm(v)*np.linalg.norm(goal):
        return obs["next_checkpoint"], 100
    else:
        return aim, 100
    
def strat2(obs):
    return {"cible" : obs["next_checkpoint"], "thrust" : 100}

def actor(obs):
    act = strat(obs)
    return {"cible" : act[0], "thrust" : act[1]}

env = (progressReward(TimeLimit(Environement(reset_checkpoints = False, render_mode="human"), 100)))
done = False

obs, _ = env.reset()

#from DDPG import Buffer

#buffer = Buffer(50000, 64)

prev_state, _ = env.reset(render_mode = "human")
episodic_reward = 0

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

while True:
    # Uncomment this to see the Actor in action
    # But not in a python notebook.


    action = strat2(prev_state)
    # Recieve state and reward from environment.

    state, reward, done,_ , info = env.step(action)

    #buffer.record((prev_state, action, reward, state))
    episodic_reward += reward

    #buffer.learn()

    # End this episode when `done` is True
    if done:
        break

    prev_state = state

ep_reward_list.append(episodic_reward)

# Mean of last 40 episodes
avg_reward = np.mean(ep_reward_list[-40:])
print("Episode * {} * Avg Reward is ==> {}".format(1, avg_reward))
avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()