import numpy as np
from matplotlib import pyplot as plt

# tworzymy tablice o wymiarach 128x128x3 (3 kanaly to RGB)
# uzupelnioną zerami = kolor czarny
data = np.zeros((128, 128), dtype=np.uint8)

# fig = plt.figure(figsize=)


# chcemy zeby obrazek byl czarnobialy,
# wiec wszystkie trzy kanaly rgb uzupelniamy tymi samymi liczbami
# napiszmy do tego funkcje
def draw(img, x, y, color):
    img[x, y] = color


# zamalowanie 4 pikseli w lewym górnym rogu
draw(data, 5, 5, 100)
draw(data, 6, 6, 100)
draw(data, 5, 6, 255)
draw(data, 6, 5, 255)


# rysowanie kilku figur na obrazku
for i in range(128):
    for j in range(128):
        if (i-64)**2 + (j-64)**2 < 900:
            draw(data, i, j, 200)
        elif i > 100 and j > 100:
            draw(data, i, j, 255)
        elif (i-15)**2 + (j-110)**2 < 25:
            draw(data, i, j, 150)
        elif (i-15)**2 + (j-110)**2 == 25 or (i-15)**2 + (j-110)**2 == 26:
            draw(data, i, j, 255)

def with_filtr(data, filtr, stride=1):
    data_ret = np.zeros((data.shape[0] - filtr.shape[0] + 1, data.shape[1] - filtr.shape[1] + 1))
    for i in range(0, data.shape[0] - filtr.shape[0] + 1, stride):
        for j in range(0, data.shape[1] - filtr.shape[1] + 1, stride):
            # print(i/stride, j/stride)
            data_ret[i//stride, j//stride] = np.sum(data[i:i+filtr.shape[0], j:j+filtr.shape[1]] * filtr)
    return data_ret


plt.imshow(data, interpolation='nearest')
plt.show()

data_b = with_filtr(data,
    np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])
, 2)

plt.imshow(data_b, interpolation='nearest')
plt.show()


data_b = with_filtr(data,
    np.array([
        [1, 1,1],
        [0,0,0],
        [-1,-1, -1]
    ])
)

plt.imshow(data_b, interpolation='nearest')
plt.show()


data_b2 = with_filtr(data,
    np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0, 1]
    ])
)

plt.imshow(data_b2, interpolation='nearest')
plt.show()

data_b3 = with_filtr(data,
    np.array([
        [0,1,2],
        [-1,0,1],
        [-2,-1, 0]
    ])
)

plt.imshow(data_b3, interpolation='nearest')
plt.show()
