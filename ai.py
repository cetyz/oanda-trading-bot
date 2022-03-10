# general concept:
# take in the inputs (TBD, last 5000 candles + whatever additional features)
# if no position open, possible actions:
#   1. buy (how much? maybe set 5 possible levels)
#   2. sell (how much?)
#   3. do nothing
#
# will be the same, but for clarity:
# if position open, possible actions:
#   1. do nothing
#   2. buy (once again 5 possible levels) (be it to close position or double down)
#   3. sell (5 possible levels)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

LEARNING_RATE = 0.00025

seed = 0 # in case we need to replicate or smth
gamma = 0.99 # discount factor for past rewards (higher gamma means more emphasis on future rewards)
epsilon = 1.0 # probability of performing random action (start at 1.0)
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_interval = epsilon_max - epsilon_min # rate at which to reduce epsilon
batch_size = 32 # size of batch taken from replay buffer
max_steps_per_episode = 17_280 # assuming 5 sec candles, this would be 24hrs (let's assume one episode is one day)

num_actions = 11 # 5 levels of buy + 5 levels of sell + do nothing

def create_q_model():

    inputs = layers.Input(shape=(5, 5000,)) # temp

    layer1 = layers.Dense(1024, activation='relu')(inputs)
    layer2 = layers.Dense(1024, activation='relu')(layer1)

    action = layers.Dense(num_actions, activation='softmax')(layer2)

    return keras.Model(inputs=inputs, outputs=action)

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)

# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model() 

