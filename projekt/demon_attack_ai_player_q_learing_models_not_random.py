import glob
import os
import time
import gymnasium
import numpy as np
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam

# Środowisko
env = gymnasium.make("ALE/DemonAttack-v5"
                     , render_mode="human"
                     )
in_dimension = env.observation_space.shape
out_dimension = env.action_space.n

# Model
def get_model():
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(210, 160, 3)))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(out_dimension, activation='linear'))  # Funkcja aktywacji linear w warstwie wyjściowej
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

model = get_model()
def get_action(observation):
    obs = np.array([observation]) / 255.0
    q_values = model.predict(obs, verbose=0)[0]
    return np.argmax(q_values)

# rozegraj grę z modelem
Model_file = "./demon_attack_q_learing_models_not_random/generator_model_000002100_656100.keras"
model.load_weights(Model_file)

observation, info = env.reset()
terminated = False
total_reward = 0

while not terminated:
    env.render()
    action = get_action(observation)
    next_observation, reward, terminated, info, _ = env.step(action)
    total_reward += reward
    observation = next_observation
    print("action:", action, "reward:", reward, "terminated:", terminated, "info:", info, "total_reward:", total_reward)
    if terminated:
        print("GAME OVER")
        print("info:", info)
        print("reward:", total_reward)
        break