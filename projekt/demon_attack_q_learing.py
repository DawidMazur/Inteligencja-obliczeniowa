import glob

import gymnasium
import numpy as np
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam
from keras.utils import to_categorical

# Środowisko
env = gymnasium.make("ALE/DemonAttack-v5", render_mode="human")
in_dimension = env.observation_space.shape
out_dimension = env.action_space.n

print("in_dimension:", in_dimension)


# konfiguracja programu
max_games_number = 1000
save_model_every = 100
model_trains = 0

# Parametry algorytmu DQN
gamma = 0.99  # Współczynnik dyskontowania
epsilon = 1.0  # Wartość epsilon dla strategii epsilon-greedy
epsilon_min = 0.01  # Minimalna wartość epsilon
epsilon_decay = 0.995  # Współczynnik zmniejszania epsilon po każdym kroku czasowym
memory = []  # Bufor pamięci dla doświadczeń (experience replay)
memory_capacity = 10000  # Pojemność bufora pamięci
batch_size = 32  # Rozmiar paczki do nauki

# Model
def get_model():
    model = Sequential()
    model.add(Dense(128, input_shape=(100800,), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(out_dimension, activation='linear'))  # Funkcja aktywacji linear w warstwie wyjściowej
    model.compile(loss='mse', optimizer=Adam(lr=0.001))  # Mean Squared Error jako funkcja kosztu
    return model

# def get_model():
#     model = Sequential()
#     model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(100800,)))
#     model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(Flatten())
#     model.add(Dense(512, activation='relu'))
#     model.add(Dense(out_dimension, activation='linear'))  # Funkcja aktywacji linear w warstwie wyjściowej
#     model.compile(loss='mse', optimizer=Adam(lr=0.001))  # Mean Squared Error jako funkcja kosztu
#     return model


# Model do nauki
model = get_model()

# Wczytanie modelu z pliku
files = glob.glob("demon_attack_q_learing_models/generator_model_*.keras")
if len(files) > 0:
    files.sort()
    model.load_weights(files[-1])
    print("Loaded generator weights from file: ", files[-1])

    model_trains = int(files[-1].split("_")[2].split(".")[0])
    if(model_trains > 300):
        epsilon = 0.01




# Funkcja wybierająca akcję zgodnie z strategią epsilon-greedy
def get_action(observation):
    if np.random.rand() <= epsilon:
        return np.random.choice(out_dimension)
    else:
        print("generowanie ruchu z modelu")
        obs_flat = observation.flatten() / 255.0
        obs_flat = np.array([obs_flat])
        q_values = model.predict(obs_flat)[0]
        return np.argmax(q_values)


# Funkcja treningowa dla algorytmu DQN
def train_dqn(model_trains):
    if len(memory) < batch_size:
        return model_trains

    print("train_dqn")
    # Losowy wybór indeksów doświadczeń z bufora pamięci
    indices = np.random.choice(len(memory), batch_size, replace=False)
    batch = [memory[i] for i in indices]

    # Sprawdzenie, czy każdy element w batchu jest tuplem
    states, actions, rewards, next_states, terminals = zip(*batch)
    # if not all(isinstance(item, tuple) for item in states):
    #     print("coś nie tak")
    #     return

    states = np.array(states) / 255.0  # Normalizacja danych wejściowych
    # każdy z 32 elementów w states zrób flatten()
    states = np.array([item.flatten() for item in states]) / 255.0
    next_states = np.array(next_states) / 255.0
    next_states = np.array([item.flatten() for item in next_states]) / 255.0

    # Obliczenie docelowej wartości Q
    targets = model.predict(states)
    targets_next = model.predict(next_states)
    for i in range(batch_size):
        targets[i, actions[i]] = rewards[i] + gamma * np.max(targets_next[i]) * (1 - terminals[i])

    model_trains += 1
    # Trenowanie modelu
    print("train_on_batch: ", model_trains)
    model.train_on_batch(states, targets)

    # Zapisanie modelu
    if model_trains % save_model_every == 0:
        filename = 'generator_model_%03d.keras' % model_trains
        model.save("demon_attack_q_learing_models/" + filename)
    return model_trains


# Nauka modelu
episodes = 1000
for episode in range(episodes):
    observation, info = env.reset()
    terminated = False
    total_reward = 0

    while not terminated:
        action = get_action(observation)
        next_observation, reward, terminated, _, _ = env.step(action)
        total_reward += reward

        # Zapisanie doświadczenia do bufora pamięci
        memory.append((observation, action, reward, next_observation, terminated))
        if len(memory) > memory_capacity:
            del memory[0]

        observation = next_observation

        # Trenowanie modelu co krok czasowy
        model_trains = train_dqn(model_trains)

        # Zmniejszenie wartości epsilon po każdym epizodzie
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print("epsilon: ", epsilon)

    print(f"Epizod: {episode + 1}, Nagroda: {total_reward}")

env.close()
