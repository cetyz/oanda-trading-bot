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

LEARNING_RATE = 0.001

num_actions = 11 # 5 levels of buy + 5 levels of sell + do nothing

def create_q_model():

    inputs = layers.Input(shape=(25000)) # temp

    layer1 = layers.Dense(1024, activation='relu')(inputs)
    layer2 = layers.Dense(1024, activation='relu')(layer1)

    action = layers.Dense(num_actions, activation='softmax')(layer2)

    return keras.Model(inputs=inputs, outputs=action)

def get_optimizer():

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)

    return(optimizer)

def get_loss_function():

    # loss_function = keras.losses.SparseCategoricalCrossentropy()
    loss_function = keras.losses.Huber()

    return(loss_function)

def get_num_actions():

    return(num_actions)

