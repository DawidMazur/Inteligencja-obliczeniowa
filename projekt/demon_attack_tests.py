import gymnasium
import numpy as np
from keras import Sequential
from keras.layers import Dense, Flatten

env = gymnasium.make("ALE/DemonAttack-v5", render_mode="human")

in_dimension = env.observation_space.shape
out_dimension = env.action_space.n

# dozwolone ruchy:
# 0 NOOP
# 1 FIRE
# 2 RIGHT
# 3 LEFT
# 4 RIGHTFIRE
# 5 LEFTFIRE

print("in_dimension:", in_dimension)
print("out_dimension:", out_dimension)


def get_model():
    model = Sequential()
    model.add(Dense(128, input_shape=(100800,), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = get_model()
model.summary()

# zmienne konfiguracyjne
MAX_STEPS_PER_GAME = 10000


def get_action_from_model(observation):
    # print("observation:", observation)
    #     convert observation to 1D array
    obs_flat = observation.flatten()
    print(observation)

    #     predict action based on observation
    action = model.predict(np.array([obs_flat]))
    # pick action with highest probability
    action = np.argmax(action)
    print("action:", action)
    return action


# nauczanie i granie
while True:
    observation, info = env.reset()
    terminated = False

    for _ in range(MAX_STEPS_PER_GAME):
        #         generate action based on observation from model
        action = get_action_from_model(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

# for _ in range(100):
#     action = env.action_space.sample()  # agent policy that uses the observation and info
#     observation, reward, terminated, truncated, info = env.step(action)
#
#     print("observation:", observation)
#     print("reward:", reward)
#     print("terminated:", terminated)
#     print("truncated:", truncated)
#     print("info:", info)
#
#     if terminated or truncated:
#         observation, info = env.reset()
#
# env.close()
