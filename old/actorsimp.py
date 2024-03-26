import tensorflow as tf
from keras import layers
import numpy as np

class Actor():

    def __init__(self, num_states, num_actions):

        self.num_states = num_states
        self.num_action = num_actions

        inputs = layers.Input(shape=(num_states,))
        hl = layers.Dense(num_states, activation = None)
        hidden = hl(inputs)
        outputs = layers.Dense(num_actions, activation="tanh")(hidden)
        self.model = tf.keras.Model(inputs, outputs)
        hl.set_weights([np.eye(num_states), np.zeros(num_states,)])

    def mutate(self, rate):
        self.set_weights([self.get_weights()[i] + np.random.normal(0, rate, self.get_weights()[i].shape) for i in range(len(self.get_weights()))])

    def copy(self):
        c = Actor(self.num_states, self.num_action)
        c.set_weights(self.get_weights())
        return c
    
    def play(self, env, render_mode = None):
        state = env.reset(render_mode = render_mode)
        done = False
        while not done:
            tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            action = np.squeeze(self.model(tf_state))
            state, done = env.step([action[0] * np.pi, action[1] * 50 + 50])    
        return env.reward()
    
    def eval(self, env, N = 1):
        scores = []
        for i in range(N):  
            scores.append(self.play(env))
        return np.mean(scores)