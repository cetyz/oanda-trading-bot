import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from ai import create_q_model, get_optimizer, get_loss_function, get_num_actions

# load in data
data = 'ETH_USD_data_2019-01-01_to_2021-12-31.csv'
df = pd.read_csv(data)
df.columns = ['datetime', 'volume', 'open', 'high', 'low', 'close']
df.datetime = pd.to_datetime(df.datetime)
df['price'] = (df['open'] + df['close']) / 2.0

# how to train?
# 
# 5000 5s candles -> 6.9 hours
# shall we let each "episode" be a two-hour period?
# 2 hours -> 1440 candles
# let's try that
# so we will randomly pick a 5000 candle date range
# simulate the pulling of 5000 candles for the next 2 hours
# i.e. move the window one candle at a time until we've moved 1440 candles. that's one episode
# then calculate the total reward obtained during that 2 hour window
# will optimize towards that reward?


# should think of a way to log training/results/improvements/etc.

# parameters
state_window = 5000

gamma = 0.99 # discount factor for past rewards (higher gamma means more emphasis on future rewards)
epsilon = 1.0 # probability of performing random action (start at 1.0)
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_interval = epsilon_max - epsilon_min # rate at which to reduce epsilon
batch_size = 32 # size of batch taken from replay buffer - about "3 mins"
# max_steps_per_episode = 17_280 # assuming 5 sec candles, this would be 24hrs (let's assume one episode is one day)
max_steps_per_episode = 1440 # 2 hours worth of 5 sec candles

living_penalty = -0.05 # penalty for choosing to do nothing (will probably need to tweak this quite a bit)


# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()

optimizer = get_optimizer()

num_actions = get_num_actions()
# 0 -> do nothing
# 1-5 -> buy size 1-5
# 6-10 -> sell size 1-5

# experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
# done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0

max_memory_length = 100000
update_after_actions = 60 # train the model every "5 mins"
update_target_network = 1440 # writing this here but in the code we will only update target network after each 2 hour episode?

loss_function = get_loss_function()

def get_state(df, start, end):

    cols = ['volume', 'open', 'high', 'low', 'close']
    state_df = df.iloc[start:end]
    state_df = state_df[cols]
    state_df = state_df.apply(lambda x: x/x.max(), axis=0)
    values = state_df.values.flatten()
    state = values

    return(state)

def get_price(df, idx):
    df_cols = df.columns.tolist()
    for i, col in enumerate(df_cols):
        if col == 'price':
            break
    price = df.iloc[idx, i]

    return(price)


# for now we need a manual way to calculate open positions
# in future the platform api will be able to calculate it for us
open_position = False
position_qty = 0
position_avg_price = 0

class Arrows:
    def __init__(self):
        self.buy_arrows =  {}
        self.sell_arrows = {}

    def add_buy(self, dt, price):        
        self.buy_arrows[dt] = price
    
    def add_sell(self, dt, price):
        self.sell_arrows[dt] = price


arrows = Arrows()

episode_rewards = []
losses = []

for i in range(20):
    
    print(f'Starting episode {i}')
    # get the random 5000 candle starting point
    
    min_start_point = 0
    max_start_point = len(df) - state_window

    random_start_point = np.random.randint(min_start_point, max_start_point)

    # definitely want to get "position open" as a feature in state but we'll do that next time
    state = get_state(df, random_start_point, random_start_point+state_window)
    episode_reward = 0

    for step in range(0, max_steps_per_episode):

        # use epsilon for exploration
        if epsilon > np.random.rand(1)[0]:
            # take random action
            action = np.random.choice(num_actions)
        else:
            # predict action q-values
            # from the environment state

            action_probs = model.predict(state.reshape(1, len(state)))
            action = np.argmax(action_probs)

        # decay probability of taking random action
        epsilon *= epsilon_interval
        epsilon = max(epsilon, epsilon_min)

        # apply the sampled action in our environment
        
        price = get_price(df, idx=random_start_point+state_window+step)

        reward = 0

        if not action:
            reward = living_penalty

        elif not open_position:
            reward = 0
            open_position = True
            if (action >= 1) and (action <= 5):
                quantity = action
                position_qty = quantity
                position_avg_price = price
            elif (action >= 6) and (action <= 10):
                quantity = action - 5
                position_qty = -quantity
                position_avg_price = price

        elif open_position or (position_qty > 0):
            # buy
            if (action >= 1) and (action <= 5):
                quantity = action
                
                # if position_qty < 0, 0, > 0

                if position_qty < 0:
                    # this means that we currently have a short position
                    # 3 possibilities
                    # 1. our order is larger than the position -> our short will close and then become a long
                    # 2. our order is == to the position -> our short will close
                    # 3. our ourder is small than the position -> our short will become smaller

                    if quantity > -position_qty:
                        # note the -ve sign because it's a short position
                        diff = (price - position_avg_price) / position_avg_price
                        reward = position_qty * diff * position_avg_price

                        # our short is closed
                        extra_qty = quantity - -position_qty

                        # new long position is opened
                        position_qty = extra_qty
                        position_avg_price = price

                    elif quantity == -position_qty:

                        diff = (price - position_avg_price) / position_avg_price
                        reward = position_qty * diff * position_avg_price

                        # position is closed
                        position_qty = 0
                        position_avg_price = 0
                        open_position = False

                    elif quantity < -position_qty:

                        diff = (price - position_avg_price) / position_avg_price
                        reward = quantity * diff * position_avg_price * -1 # note the -1

                        position_qty = position_qty + quantity
                        # position_avg price remains the same

                elif not position_qty: # i.e. no open position (code should not actually reach here, but just in case)
                    
                    position_qty = quantity
                    position_avg_price = price
                    reward = 0

                elif position_qty > 0:
                    # i.e. we are already in a long position. now just adding to it

                    old_position_cost = position_qty * position_avg_price
                    new_position_cost = quantity * price
                    position_qty = position_qty + quantity
                    position_avg_price = (old_position_cost + new_position_cost) / position_qty

                    reward = 0



            # sell
            elif (action >= 6) and (action <= 10):
                quantity = action - 5
                
                # if position_qty < 0, 0, > 0

                if position_qty < 0:
                    # it means we already have a short position. just need to add to it

                    old_position_cost = position_qty * position_avg_price * -1
                    new_position_cost = quantity * price
                    position_qty = -position_qty + quantity
                    position_avg_price = (old_position_cost + new_position_cost) / -position_qty

                    reward = 0

                elif not position_qty: # just to catch, in case

                    position_qty = -quantity
                    position_avg_price = price

                    reward = 0

                elif position_qty > 0:
                    # this means that we currently have a long position
                    # 3 possibilities
                    # 1. our order is larger than the position -> our long will close and then become a short
                    # 2. our order is == to the position -> our long will close
                    # 3. our order is smaller than the position -> our long will become smaller

                    if quantity > position_qty:
                        diff = (price - position_avg_price) / position_avg_price
                        reward = position_qty * diff * position_avg_price

                        # our short is closed
                        extra_qty = quantity - position_qty

                        # new short position is opened
                        position_qty = -extra_qty
                        position_avg_price = price        

                    elif quantity == position_qty:

                        diff = (price - position_avg_price) / position_avg_price
                        reward = position_qty * diff * position_avg_price

                        # position is closed
                        position_qty = 0
                        position_avg_price = 0
                        open_position = False

                    elif quantity < position_qty:
                        diff = (price - position_avg_price) / position_avg_price
                        reward = quantity * diff * position_avg_price 

                        position_qty = position_qty - quantity                                                                

        state_next = get_state(df, random_start_point+step, random_start_point+state_window+step)

        episode_reward += reward

        # save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        rewards_history.append(reward)
        state = state_next

        # some arrow stuff
        df_cols = df.columns.tolist()
        for j, col in enumerate(df_cols):
            if col == 'datetime':
                break
        if (action >= 1) and (action <= 5):
            arrows.add_buy(df.iloc[random_start_point+state_window+step, j], price)
            if len(arrows.buy_arrows) > 3:
                arrows.buy_arrows.popitem()
        elif (action >= 6) and (action <= 10):
            arrows.add_sell(df.iloc[random_start_point+state_window+step, j], price)
            if len(arrows.sell_arrows) > 3:
                arrows.sell_arrows.popitem()




        if step % update_after_actions == 0 and len(action_history) > batch_size:

            # get random indices to sample from replay buffers
            indices = np.random.choice(range(len(action_history)), size=batch_size)

            # using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample)

            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)
                losses.append(loss)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # if step % update_target_network == 0:
        #     # update the the target network with new weights
        #     model_target.set_weights(model.get_weights())
        
        #####
        # gonna chuck some visualisation in here
        #####
        
        # viz_window = 540
        # temp_df = df.iloc[random_start_point+state_window-viz_window+step:random_start_point+state_window+step]

        # plt.clf()
        # plt.plot(temp_df['datetime'], temp_df['price'])

        # # arrow should be say 5% of the chart height
        # y_min = temp_df['price'].min()
        # y_max = temp_df['price'].max()
        # arrow_height = (y_max - y_min) * 0.05

        # for arrow_x, arrow_y in arrows.buy_arrows.items():
        #     plt.arrow(arrow_x, arrow_y-arrow_height*2, 0, arrow_height, color='green')
        # for arrow_x, arrow_y in arrows.sell_arrows.items():
        #     plt.arrow(arrow_x, arrow_y+arrow_height*2, 0, -arrow_height, color='red')

        # plt.pause(0.05)

        #######
        #######
        #######
        #         
        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]

    model_target.set_weights(model.get_weights())

    print(f'Episode {i} reward: {episode_reward}')
    episode_count += 1
    episode_rewards.append(episode_reward)

print(episode_rewards)

plt.clf()
plt.plot(range(len(losses)), losses)
plt.show()

# visualize

# class Arrows:
#     def __init__(self):
#         self.buy_arrows =  {}
#         self.sell_arrows = {}

#     def add_buy(self, dt, price):        
#         self.buy_arrows[dt] = price
    
#     def add_sell(self, dt, price):
#         self.sell_arrows[dt] = price


# arrows = Arrows()

# # arrow_date1 = df.iloc[100].datetime
# # arrow_date2 = df.iloc[150].datetime
# # arrow_price1 = df.loc[df.datetime==arrow_date1, 'price'].values[0]
# # arrow_price2 = df.loc[df.datetime==arrow_date2, 'price'].values[0]

# # arrows.add_buy(arrow_date1, arrow_price1)
# # arrows.add_sell(arrow_date2, arrow_price2)

# window = 720

# for i in range(0, len(df)):
    
#     temp_df = df.iloc[i:i+window]

#     plt.clf()
#     plt.plot(temp_df['datetime'], temp_df['price'])

#     # arrow should be say 5% of the chart height
#     y_min = temp_df['price'].min()
#     y_max = temp_df['price'].max()
#     arrow_height = (y_max - y_min) * 0.05

#     for arrow_x, arrow_y in arrows.buy_arrows.items():
#         plt.arrow(arrow_x, arrow_y-arrow_height*2, 0, arrow_height, color='green')
#     for arrow_x, arrow_y in arrows.sell_arrows.items():
#         plt.arrow(arrow_x, arrow_y+arrow_height*2, 0, -arrow_height, color='green')

#     plt.pause(0.5)

# plt.show()
