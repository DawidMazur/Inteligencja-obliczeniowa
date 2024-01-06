import glob
import os
import time
import gymnasium
import numpy as np
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam

import logging



# Środowisko
env = gymnasium.make("ALE/DemonAttack-v5"
                     # , render_mode="human"
                     )
in_dimension = env.observation_space.shape
out_dimension = env.action_space.n

print("in_dimension:", in_dimension)


# konfiguracja programu
max_games_number = 10000
save_model_every = 300
model_trains = 0
start_time = time.time()

# version_name = "demon_not_random_learing"
version_name = "q_learing_models_not_random"

logging.basicConfig(filename='log_' + version_name + '.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parametry algorytmu DQN
gamma = 0.99  # Współczynnik dyskontowania
epsilon = 1.0  # Wartość epsilon dla strategii epsilon-greedy
epsilon_min = 0.01  # Minimalna wartość epsilon
epsilon_decay = 0.900  # Współczynnik zmniejszania epsilon po każdym kroku czasowym
memory = []  # Bufor pamięci dla doświadczeń (experience replay)
memory_capacity = 1000  # Pojemność bufora pamięci
batch_size = 32  # Rozmiar paczki do nauki

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


# Model do nauki
model = get_model()

# Wczytanie modelu z pliku
files = glob.glob("demon_attack_" + version_name + "/generator_model_*.keras")
if len(files) > 0:
    files.sort()
    model.load_weights(files[-1])
    print("Loaded generator weights from file: ", files[-1])

    last_file_name = files[-1].split("/")[-1]
    print("last_file_name: ", last_file_name)

    model_trains = int(last_file_name.split("_")[2])
    epsilon = float(last_file_name.split("_")[3].split(".")[0]) / 1000000




# Funkcja wybierająca akcję zgodnie z strategią epsilon-greedy
def get_action(observation):
    if np.random.rand() <= epsilon:
        return np.random.choice(out_dimension)
    else:
        obs = np.array([observation]) / 255.0
        q_values = model.predict(obs, verbose=0)[0]
        move = np.argmax(q_values)
        print("generowanie ruchu z modelu:", str(move))
        return move


# Funkcja treningowa dla algorytmu DQN
def train_dqn():
    if len(memory) < batch_size:
        return

    global model_trains
    global epsilon
    global save_model_every

    # wybierz ostatnie doświadczenia
    indices = range(len(memory) - batch_size, len(memory))
    batch = [memory[i] for i in indices]

    states, actions, rewards, next_states, terminals, total_rewards = zip(*batch)

    states = np.array(states) / 255.0
    next_states = np.array(next_states) / 255.0

    # Obliczenie docelowej wartości Q
    targets = model.predict(states, verbose=0)
    targets_next = model.predict(next_states, verbose=0)
    for i in range(batch_size):
        targets[i, actions[i]] = rewards[i] + gamma * np.max(targets_next[i]) * (1 - terminals[i])

    model_trains += 1
    # Trenowanie modelu
    print("train_on_batch: ", model_trains)
    model.train_on_batch(states, targets)

    # Zapisanie modelu
    if model_trains % save_model_every == 0:
        epsilon_str = str(epsilon * 1000000).split(".")[0]
        filename = 'generator_model_%09d_' % model_trains
        filename = filename + epsilon_str + '.keras'
        model.save("demon_attack_" + version_name + "/" + filename)
        global start_time
        save_model_every_str = str(save_model_every)
        time_str = str(time.time() - start_time)
        print("ostatenie " + save_model_every_str + " zajeło: ", time_str)
        start_time = time.time()

#         usuń stare modele, zostaw 5 najnowszych
#         files = glob.glob("demon_attack_q_learing_models/generator_model_*.keras")
#         if len(files) > 5:
#             files.sort()
#             for i in range(len(files) - 5):
#                 os.remove(files[i])
#                 print("usunięto plik: ", files[i])


# Nauka modelu
for episode in range(max_games_number):
    observation, info = env.reset()
    terminated = False
    total_reward = 0

    while not terminated:
        action = get_action(observation)
        next_observation, reward, terminated, _, _ = env.step(action)
        total_reward += reward

        # Zapisanie doświadczenia do bufora pamięci
        memory.append((observation, action, reward, next_observation, terminated, total_reward))
        if len(memory) > memory_capacity:
            del memory[0]

        observation = next_observation

        # Trenowanie modelu co krok czasowy
        train_dqn()

    # Zmniejszenie wartości epsilon po każdym epizodzie
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print("epsilon: ", epsilon)

    print('*' * 50)
    print('*' * 50)
    print('*' * 50)
    print(f"Epizod: {episode + 1}, Nagroda: {total_reward}")
    epsilon_str = str(epsilon)
    model_trains_str = str(model_trains)
    total_reward_str = str(total_reward)
    episode_str = str(episode+1)
    logging.info("Epizod: " + episode_str + " Nagroda: " + total_reward_str + " epsilon: " + epsilon_str + " model_trains: " + model_trains_str)
    print('*' * 50)
    print('*' * 50)
    print('*' * 50)

env.close()
