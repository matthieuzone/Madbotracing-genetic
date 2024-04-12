import tensorflow as tf
from keras import layers
import numpy as np

def create_model(num_states, num_actions):
    inputs = layers.Input(shape=(num_states,))
    outputs = layers.Dense(num_actions, activation="tanh")(inputs)
    return tf.keras.Model(inputs, outputs)

def addhl(model, n):
    num_states = model.input.shape[1]
    num_actions = model.output.shape[1]

    inputs = layers.Input(shape=(num_states,))
    hl = layers.Dense(n, activation="relu")(inputs)
    con = layers.Concatenate()([inputs, hl])
    ol = layers.Dense(num_actions, activation="tanh")
    outputs = ol(con)        
    model2 =  tf.keras.Model(inputs, outputs)

    ol.set_weights([np.concatenate([ model.get_weights()[0], np.zeros((n,num_actions))], axis = 0) , model.get_weights()[1]])
    return model2

def addinputs(model, n):
    num_states = model.input.shape[1] + n
    num_actions = model.output.shape[1]

    inputs = layers.Input(shape=(num_states,))
    outputs =  layers.Dense(num_actions, activation="tanh")(inputs)
    model2 =  tf.keras.Model(inputs, outputs)

    model2.set_weights([np.concatenate([ model.get_weights()[0], np.zeros((n,num_actions))], axis = 0),  model.get_weights()[1]])

    return model2

class Actor():

    def __init__(self, model):
        self.model = model

    def mutate(self, rate):
        self.model.set_weights([self.model.get_weights()[i] + np.random.normal(0, rate, self.model.get_weights()[i].shape) for i in range(len(self.model.get_weights()))])

    def copy(self):
        c = Actor(tf.keras.models.clone_model(self.model))
        c.model.set_weights(self.model.get_weights())
        return c
    
    def play(self, env):
        state, _ = env.reset()
        done = False
        while not done:
            tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            action = np.squeeze(self.model(tf_state))
            state, _, done, _, _ = env.step(action)
        return env.score()
    
    def eval(self, env, N = 1):
        scores = []
        for i in range(N):
            scores.append(self.play(env))
        return np.mean(scores)
