# Jaki jest plan działania:
# a) Zapoznaj się z wybranym samouczkiem np.
# a. keras:
# i. https://machinelearningmastery.com/how-to-develop-aconvolutional-neural-network-to-classify-photos-of-dogs-and-cats/
# ii. https://www.kaggle.com/code/uysimty/keras-cnn-dog-or-catclassification
# b. tensorflow:
# i. https://pythonprogramming.net/convolutional-neural-networkkats-vs-dogs-machine-learning-tutorial/
# ii. (można wyszukać inne)
# b) Załaduj bazę danych i dokonaj jej obróbki (przetworzenie obrazów, wyciągnięcie
# klasy cat/dog z nazwy obrazka).
# c) Podziel dane na zbiór testowy i treningowy, i być może też walidacyjny.
# d) Skonstruuj, wytrenuj model sieci konwolucyjnej na zbiorze treningowym. Sieć
# może mieć standardową, zaproponowaną w Internecie konfigurację.
# e) Podaj krzywą uczenia się dla zbioru treningowego i walidacyjnego.
# f) Podaj dokładność sieci na zbiorze testowym.
# g) Podaj macierz błędu na zbiorze testowym.
# (Pytania dodatkowe: Czy są koty przypominające psy, albo psy przypominające
# koty? Ile? Potrafisz może wskazać konkretne zdjęcia omyłkowo
# zakwalifikowanych zwierząt?)
# h) Powtórz parę razy eksperyment d-g z inną konfiguracją sieci (optimizer, funkcje
# aktywacji, inna struktura sieci, dropout, itp.). Wskaż jaka konfiguracja działała
# najlepiej i pokaż jej wyniki (krzywa uczenia się, dokładność, macierz błędu
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

FILE_PATH = './dogs-cats-mini/'
IMG_SIZE = 50


def prepare_data():
    prepered_data = []
    for img in tqdm(os.listdir(FILE_PATH)):
        label = img.split('.')[0]
        if label == 'cat':
            label = 1
        elif label == 'dog':
            label = 0
        path = os.path.join(FILE_PATH, img)

        # zmiana rozmiaru obrazka i konwersja na odcienie szarosci
        img = cv2.resize(
            cv2.imread(path, cv2.IMREAD_GRAYSCALE),
            (IMG_SIZE, IMG_SIZE)
        )
        prepered_data.append([np.array(img), np.array(label)])
    shuffle(prepered_data)
    return prepered_data


# 2. załadowanie danych
data = prepare_data()
# 3. testowy i treningowy
train = data[:-500]
test = data[-500:]

# 4. konstrukcja sieci
# define cnn model
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]
model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500, show_metric=True)
